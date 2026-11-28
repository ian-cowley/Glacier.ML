using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;
using Glacier.Gpu.RingBuffer;
using Glacier.ML.Core;

namespace Glacier.ML.Compute;

/// <summary>
/// Hardware accelerator for Glacier.ML operations.
/// Delivers bare-metal NVIDIA CUDA offload for K-Means cluster assignment and persistent ring-buffer
/// vector kernels (<see cref="VectorAdd"/> / <see cref="VectorFma"/>), and multi-core CPU SIMD for
/// <see cref="ComputeGramMatrix"/> and <see cref="ProjectFeatures"/>.
/// </summary>
public static unsafe class GpuMlAccelerator
{
    private const int RingBufferAutoThreshold = 65536;
    private static readonly Lock s_initLock = new();
    private static readonly Lock s_ringBufferLock = new();
    private static readonly System.Collections.Concurrent.ConcurrentQueue<GpuMlStreamContext> s_streamPool = new();

    private static GpuMlStreamContext RentContext()
    {
        if (s_streamPool.TryDequeue(out var ctx))
            return ctx;
        return new GpuMlStreamContext();
    }

    private static void ReturnContext(GpuMlStreamContext ctx)
    {
        s_streamPool.Enqueue(ctx);
    }
    private static bool s_nvidiaInitialized;
    private static bool s_nvidiaAvailable;
    private static IntPtr s_cuContext;
    private static IntPtr s_cuModule;
    private static IntPtr s_fnKMeansAssign;


    // Persistent Megakernel Ring Buffer engine & unified device-mapped memory
    private static NvidiaSassEngine? s_nvidiaEngine;
    private static PersistentRingBuffer? s_ringBuffer;
    private static IntPtr s_hVecA;
    private static IntPtr s_dVecA;
    private static IntPtr s_hVecB;
    private static IntPtr s_dVecB;
    private static IntPtr s_hVecC;
    private static IntPtr s_dVecC;
    private static nuint s_capVecBytes;

    static GpuMlAccelerator()
    {
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => ShutdownRingBuffer();
    }

    internal static IntPtr ContextHandle => s_cuContext;

    public static bool IsNvidiaAvailable => EnsureNvidiaInitialized();
    public static bool IsAmdAvailable => false;
    public static bool IsGpuAvailable => IsNvidiaAvailable;

    #region Driver Initialization

    private static bool EnsureNvidiaInitialized()
    {
        if (s_nvidiaInitialized) return s_nvidiaAvailable;
        lock (s_initLock)
        {
            if (s_nvidiaInitialized) return s_nvidiaAvailable;
            try
            {
                if (!CuDriver.IsAvailable())
                {
                    s_nvidiaAvailable = false;
                    s_nvidiaInitialized = true;
                    return false;
                }

                if (CuDriver.Init(0) != 0)
                {
                    s_nvidiaAvailable = false;
                    s_nvidiaInitialized = true;
                    return false;
                }

                if (CuDriver.DeviceGet(out int dev, 0) != 0)
                {
                    s_nvidiaAvailable = false;
                    s_nvidiaInitialized = true;
                    return false;
                }

                // Check compute capability
                if (CuDriver.DeviceGetAttribute(out int major, 75, dev) != 0 ||
                    CuDriver.DeviceGetAttribute(out int minor, 76, dev) != 0)
                {
                    s_nvidiaAvailable = false;
                    return false;
                }
                string targetArch = $"sm_{major}{minor}";

                if (CuDriver.CtxCreate(out s_cuContext, 0, dev) != 0)
                {
                    s_cuContext = IntPtr.Zero;
                    s_nvidiaAvailable = false;
                    return false;
                }

                string ptx = KernelCompiler.NormalizePtxDirectives(GpuMlKernels.PtxSource, targetArch);

                byte[] ptxBytes = Encoding.UTF8.GetBytes(ptx + "\0");
                if (CuDriver.ModuleLoadData(out s_cuModule, ptxBytes) != 0)
                {
                    s_cuModule = IntPtr.Zero;
                    CleanupNvidiaInitState();
                    s_nvidiaAvailable = false;
                    return false;
                }

                if (CuDriver.ModuleGetFunction(out s_fnKMeansAssign, s_cuModule, "kmeans_assign_fp32") != 0)
                {
                    s_fnKMeansAssign = IntPtr.Zero;
                    CleanupNvidiaInitState();
                    s_nvidiaAvailable = false;
                    return false;
                }

                s_nvidiaAvailable = true;
            }
            catch
            {
                CleanupNvidiaInitState();
                s_nvidiaAvailable = false;
            }
            finally
            {
                s_nvidiaInitialized = true;
            }

            return s_nvidiaAvailable;
        }
    }

    private static void CleanupNvidiaInitState()
    {
        if (s_cuModule != IntPtr.Zero)
        {
            CuDriver.ModuleUnload(s_cuModule);
            s_cuModule = IntPtr.Zero;
        }
        if (s_cuContext != IntPtr.Zero)
        {
            CuDriver.CtxDestroy(s_cuContext);
            s_cuContext = IntPtr.Zero;
        }
        s_fnKMeansAssign = IntPtr.Zero;
    }

    #endregion

    #region K-Means Assignment Offload

    /// <summary>
    /// Executes nearest-centroid assignment across N samples on the specified hardware target.
    /// Returns true if executed on NVIDIA GPU, false if caller should use CPU fallback.
    /// Throws <see cref="NotSupportedException"/> for unsupported targets (<see cref="GpuTarget.Amd"/>, <see cref="GpuTarget.DualGpu"/>).
    /// </summary>
    public static bool AssignClustersGpu(
        ReadOnlySpan<float> samples,
        ReadOnlySpan<float> centroids,
        Span<int> assignments,
        int n,
        int k,
        int d,
        GpuTarget target = GpuTarget.Auto)
    {
        if (target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{target} is not supported by GpuMlAccelerator.AssignClustersGpu. Use Auto, Nvidia, or Cpu.");

        if (n <= 0 || k <= 0 || d <= 0) return false;

        switch (target)
        {
            case GpuTarget.Cpu:
                return false;

            case GpuTarget.Nvidia:
                return ExecuteNvidiaKMeansAssign(samples, centroids, assignments, n, k, d);

            case GpuTarget.Auto:
                // For small datasets (< 512 rows), CPU AVX-512 is fast enough without PCIe round-trip
                if (n < 512) return false;

                if (EnsureNvidiaInitialized() && ExecuteNvidiaKMeansAssign(samples, centroids, assignments, n, k, d))
                    return true;

                return false;

            default:
                throw new NotSupportedException($"GpuTarget.{target} is not supported.");
        }
    }

    private static bool ExecuteNvidiaKMeansAssign(
        ReadOnlySpan<float> samples,
        ReadOnlySpan<float> centroids,
        Span<int> assignments,
        int n,
        int k,
        int d)
    {
        if (!EnsureNvidiaInitialized()) return false;

        long sampleElements = (long)n * d;
        long centroidElements = (long)k * d;
        // kmeans_assign_fp32 PTX computes 32-bit offsets via mul.lo.s32
        if (sampleElements > int.MaxValue || centroidElements > int.MaxValue)
            return false;
        if (samples.Length < sampleElements || centroids.Length < centroidElements || assignments.Length < n)
            return false;

        nuint bytesSamples = checked((nuint)sampleElements * sizeof(float));
        nuint bytesCentroids = checked((nuint)centroidElements * sizeof(float));
        nuint bytesAssignments = checked((nuint)(uint)n * sizeof(int));

        GpuMlStreamContext ctx = RentContext();
        try
        {
            if (ctx.Stream == IntPtr.Zero)
                return false;
            if (CuDriver.CtxSetCurrent(s_cuContext) != 0)
                return false;

            ctx.EnsureCapacity(bytesSamples, bytesCentroids, bytesAssignments);
            if (ctx.DeviceSamples == IntPtr.Zero || ctx.DeviceCentroids == IntPtr.Zero || ctx.DeviceAssignments == IntPtr.Zero)
                return false;

            fixed (float* pSamples = samples)
            fixed (float* pCentroids = centroids)
            fixed (int* pAssignments = assignments)
            {
                if (CuDriver.MemcpyHtoDAsync(ctx.DeviceSamples, (IntPtr)pSamples, bytesSamples, ctx.Stream) != 0)
                    return false;
                if (CuDriver.MemcpyHtoDAsync(ctx.DeviceCentroids, (IntPtr)pCentroids, bytesCentroids, ctx.Stream) != 0)
                    return false;

                IntPtr argSamples = ctx.DeviceSamples;
                IntPtr argCentroids = ctx.DeviceCentroids;
                IntPtr argAssignments = ctx.DeviceAssignments;
                int argN = n;
                int argK = k;
                int argD = d;

                void** pKernelParams = stackalloc void*[6];
                pKernelParams[0] = &argSamples;
                pKernelParams[1] = &argCentroids;
                pKernelParams[2] = &argAssignments;
                pKernelParams[3] = &argN;
                pKernelParams[4] = &argK;
                pKernelParams[5] = &argD;

                uint blockSize = 256;
                uint gridSize = (uint)(((long)n + 255) / 256);

                int launchRes = CuDriver.LaunchKernel(
                    s_fnKMeansAssign,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, ctx.Stream,
                    (IntPtr)pKernelParams,
                    IntPtr.Zero);

                if (launchRes != 0) return false;

                if (CuDriver.MemcpyDtoHAsync((IntPtr)pAssignments, ctx.DeviceAssignments, bytesAssignments, ctx.Stream) != 0)
                    return false;
                if (CuDriver.StreamSynchronize(ctx.Stream) != 0)
                    return false;

                return true;
            }
        }
        catch
        {
            return false;
        }
        finally
        {
            ReturnContext(ctx);
        }
    }

    #endregion

    #region Matrix Operations (CPU SIMD Gram Matrix / Projection)

    /// <summary>
    /// Computes the Gram matrix G = X^T * X (size D x D) from feature matrix X (size N x D) on multi-core CPU SIMD.
    /// Supports <see cref="GpuTarget.Auto"/> and <see cref="GpuTarget.Cpu"/>; throws <see cref="NotSupportedException"/> for GPU-only targets.
    /// </summary>
    public static void ComputeGramMatrix(FeatureMatrix x, Span<float> gram, GpuTarget target = GpuTarget.Auto)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (target is GpuTarget.Nvidia or GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{target} is not supported by ComputeGramMatrix (CPU SIMD only). Use Auto or Cpu.");

        int n = x.Rows;
        int d = x.Columns;
        long dd = (long)d * d;
        if (gram.Length < dd)
            throw new ArgumentException("Gram buffer is too small.", nameof(gram));
        if (n == 0 || d == 0) return;

        if (x.IsColumnar)
        {
            ReadOnlyMemory<float>[] colMemories = new ReadOnlyMemory<float>[d];
            for (int c = 0; c < d; c++) colMemories[c] = x.GetColumnMemory(c);

            fixed (float* pGram = gram)
            {
                float* pG = pGram;
                Parallel.For(0, d, i =>
                {
                    ReadOnlySpan<float> colI = colMemories[i].Span;
                    for (int j = i; j < d; j++)
                    {
                        float val = DotProductSimd(colI, colMemories[j].Span);
                        pG[(long)i * d + j] = val;
                        pG[(long)j * d + i] = val;
                    }
                });
            }
            return;
        }

        // Transpose X into X_T (size D x N) using 64-byte aligned unmanaged scratchpad to avoid LOH fragmentation
        nuint byteCount = checked((nuint)((long)d * n * sizeof(float)));
        float* pXTransposed = (float*)NativeMemory.AlignedAlloc(byteCount, 64);

        try
        {
            fixed (float* pXRaw = x.Span)
            {
                float* pSrc = pXRaw;
                Parallel.For(0, d, col =>
                {
                    long colOffset = (long)col * n;
                    for (int row = 0; row < n; row++)
                    {
                        pXTransposed[colOffset + row] = pSrc[(long)row * d + col];
                    }
                });
            }

            // Symmetric Gram calculation: G[i, j] = G[j, i] = dot(col_i, col_j)
            fixed (float* pGram = gram)
            {
                float* pG = pGram;
                Parallel.For(0, d, i =>
                {
                    var colI = new ReadOnlySpan<float>(pXTransposed + ((long)i * n), n);
                    for (int j = i; j < d; j++)
                    {
                        var colJ = new ReadOnlySpan<float>(pXTransposed + ((long)j * n), n);
                        float val = DotProductSimd(colI, colJ);
                        pG[(long)i * d + j] = val;
                        pG[(long)j * d + i] = val;
                    }
                });
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pXTransposed);
        }
    }

    /// <summary>
    /// Computes feature projection Y = X * W where X is (N x D) and W is (D x K), producing (N x K) on multi-core CPU SIMD.
    /// Supports <see cref="GpuTarget.Auto"/> and <see cref="GpuTarget.Cpu"/>; throws <see cref="NotSupportedException"/> for GPU-only targets.
    /// </summary>
    public static void ProjectFeatures(
        FeatureMatrix x,
        ReadOnlySpan<float> weights,
        Span<float> destination,
        int k,
        GpuTarget target = GpuTarget.Auto)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (target is GpuTarget.Nvidia or GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{target} is not supported by ProjectFeatures (CPU SIMD only). Use Auto or Cpu.");
        if (k <= 0)
            throw new ArgumentOutOfRangeException(nameof(k), "Projection component count k must be positive.");

        int n = x.Rows;
        int d = x.Columns;
        if (weights.Length < (long)d * k)
            throw new ArgumentException("Weights buffer length mismatch.", nameof(weights));
        if (destination.Length < (long)n * k)
            throw new ArgumentException("Destination buffer length mismatch.", nameof(destination));
        if (n == 0 || d == 0) return;

        if (x.IsColumnar)
        {
            ReadOnlyMemory<float>[] colMemories = new ReadOnlyMemory<float>[d];
            for (int c = 0; c < d; c++) colMemories[c] = x.GetColumnMemory(c);

            fixed (float* pW = weights)
            fixed (float* pDest = destination)
            {
                float* pWPtr = pW;
                float* pD = pDest;
                Parallel.For(0, n, i =>
                {
                    long yOffset = (long)i * k;
                    for (int j = 0; j < k; j++)
                    {
                        pD[yOffset + j] = 0f;
                    }
                    for (int p = 0; p < d; p++)
                    {
                        float xp = colMemories[p].Span[i];
                        long wRow = (long)p * k;
                        for (int j = 0; j < k; j++)
                        {
                            pD[yOffset + j] += xp * pWPtr[wRow + j];
                        }
                    }
                });
            }
            return;
        }

        // Transpose weights so each component's weights of length D are contiguous in memory: W_T (K x D)
        nuint byteCount = checked((nuint)((long)k * d * sizeof(float)));
        float* pWTransposed = (float*)NativeMemory.AlignedAlloc(byteCount, 64);

        try
        {
            for (int p = 0; p < d; p++)
            {
                long pRow = (long)p * k;
                for (int j = 0; j < k; j++)
                {
                    pWTransposed[(long)j * d + p] = weights[(int)(pRow + j)];
                }
            }

            fixed (float* pXRaw = x.Span)
            fixed (float* pDest = destination)
            {
                float* pSrc = pXRaw;
                float* pD = pDest;
                Parallel.For(0, n, i =>
                {
                    var xRow = new ReadOnlySpan<float>(pSrc + ((long)i * d), d);
                    long yOffset = (long)i * k;
                    for (int j = 0; j < k; j++)
                    {
                        var wComp = new ReadOnlySpan<float>(pWTransposed + ((long)j * d), d);
                        pD[yOffset + j] = DotProductSimd(xRow, wComp);
                    }
                });
            }
        }
        finally
        {
            NativeMemory.AlignedFree(pWTransposed);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static float DotProductSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        int i = 0;
        int length = a.Length;

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var acc512 = Vector512<float>.Zero;
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(a), (nuint)i);
                var vb = Vector512.LoadUnsafe(ref MemoryMarshal.GetReference(b), (nuint)i);
                acc512 = Vector512.FusedMultiplyAdd(va, vb, acc512);
                i += step;
            }
            sum += Vector512.Sum(acc512);
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var acc256 = Vector256<float>.Zero;
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a), (nuint)i);
                var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b), (nuint)i);
                acc256 = Vector256.FusedMultiplyAdd(va, vb, acc256);
                i += step;
            }
            sum += Vector256.Sum(acc256);
        }

        for (; i < length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    #endregion

    #region Persistent Ring Buffer Vector Operations

    /// <summary>
    /// Gets or lazily initializes the persistent GPU ring buffer worker.
    /// </summary>
    public static PersistentRingBuffer? GetRingBuffer()
    {
        if (s_ringBuffer != null && !s_ringBuffer.IsDisposed) return s_ringBuffer;
        lock (s_initLock)
        {
            if (s_ringBuffer != null && !s_ringBuffer.IsDisposed) return s_ringBuffer;
            if (!EnsureNvidiaInitialized()) return null;

            try
            {
                s_nvidiaEngine ??= new NvidiaSassEngine();
                if (!s_nvidiaEngine.IsInitialized)
                    s_nvidiaEngine.Initialize();

                s_ringBuffer = s_nvidiaEngine.GetOrCreateRingBuffer(256);
                return s_ringBuffer;
            }
            catch
            {
                return null;
            }
        }
    }

    /// <summary>
    /// Cleanly shuts down the persistent GPU ring buffer worker if running.
    /// </summary>
    public static void ShutdownRingBuffer()
    {
        lock (s_ringBufferLock)
        {
            try
            {
                if (s_ringBuffer != null && !s_ringBuffer.IsDisposed)
                {
                    s_ringBuffer.Dispose();
                    s_ringBuffer = null;
                }
            }
            catch
            {
                // Ignore driver teardown errors during process exit
            }
        }
    }

    private static bool EnsureVectorBuffers(nuint requiredBytes)
    {
        nuint allocBytes = requiredBytes < 1024 * 1024 ? 1024 * 1024 : requiredBytes;
        if (allocBytes <= s_capVecBytes && s_hVecA != IntPtr.Zero && s_hVecB != IntPtr.Zero && s_hVecC != IntPtr.Zero)
            return true;

        // cuMemFreeHost is a synchronous context barrier; stop the persistent worker before freeing host-mapped memory.
        bool wasRingRunning = s_ringBuffer != null && !s_ringBuffer.IsDisposed;
        if (wasRingRunning)
        {
            s_ringBuffer!.Dispose();
            s_ringBuffer = null;
        }

        if (s_hVecA != IntPtr.Zero) { CuDriver.MemFreeHost(s_hVecA); s_hVecA = s_dVecA = IntPtr.Zero; }
        if (s_hVecB != IntPtr.Zero) { CuDriver.MemFreeHost(s_hVecB); s_hVecB = s_dVecB = IntPtr.Zero; }
        if (s_hVecC != IntPtr.Zero) { CuDriver.MemFreeHost(s_hVecC); s_hVecC = s_dVecC = IntPtr.Zero; }
        s_capVecBytes = 0;

        uint flags = CuDriver.CU_MEMHOSTALLOC_DEVICEMAP | CuDriver.CU_MEMHOSTALLOC_PORTABLE;
        IntPtr hA = IntPtr.Zero, dA = IntPtr.Zero;
        IntPtr hB = IntPtr.Zero, dB = IntPtr.Zero;
        IntPtr hC = IntPtr.Zero, dC = IntPtr.Zero;

        if (CuDriver.MemHostAlloc(out hA, allocBytes, flags) != 0 ||
            CuDriver.MemHostGetDevicePointer(out dA, hA, 0) != 0 ||
            CuDriver.MemHostAlloc(out hB, allocBytes, flags) != 0 ||
            CuDriver.MemHostGetDevicePointer(out dB, hB, 0) != 0 ||
            CuDriver.MemHostAlloc(out hC, allocBytes, flags) != 0 ||
            CuDriver.MemHostGetDevicePointer(out dC, hC, 0) != 0)
        {
            if (hA != IntPtr.Zero) CuDriver.MemFreeHost(hA);
            if (hB != IntPtr.Zero) CuDriver.MemFreeHost(hB);
            if (hC != IntPtr.Zero) CuDriver.MemFreeHost(hC);
            return false;
        }

        s_hVecA = hA;
        s_dVecA = dA;
        s_hVecB = hB;
        s_dVecB = dB;
        s_hVecC = hC;
        s_dVecC = dC;

        s_capVecBytes = allocBytes;
        if (wasRingRunning && s_nvidiaEngine != null)
        {
            s_ringBuffer = s_nvidiaEngine.GetOrCreateRingBuffer(256);
        }

        return true;
    }

    /// <summary>
    /// Executes element-wise vector addition: C = A + B.
    /// Dispatches to the persistent GPU ring buffer when available for large vectors (&gt;= 65536 under Auto)
    /// or explicit <see cref="GpuTarget.Nvidia"/>, otherwise executes hardware SIMD.
    /// Throws <see cref="NotSupportedException"/> for unsupported targets (<see cref="GpuTarget.Amd"/>, <see cref="GpuTarget.DualGpu"/>).
    /// </summary>
    public static void VectorAdd(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> destination,
        GpuTarget target = GpuTarget.Auto)
    {
        if (target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{target} is not supported by GpuMlAccelerator.VectorAdd. Use Auto, Nvidia, or Cpu.");

        int count = a.Length;
        if (b.Length != count || destination.Length < count)
            throw new ArgumentException("Vector dimension mismatch.");

        if (count == 0) return;

        bool useGpu = target switch
        {
            GpuTarget.Cpu => false,
            GpuTarget.Nvidia => true,
            GpuTarget.Auto => count >= RingBufferAutoThreshold && IsNvidiaAvailable,
            _ => throw new NotSupportedException($"GpuTarget.{target} is not supported.")
        };

        if (useGpu && ExecuteRingBufferVectorOp(TaskOpCode.VectorAdd, a, b, destination, 0f))
            return;

        VectorAddSimd(a, b, destination);
    }

    /// <summary>
    /// Executes element-wise fused multiply-add: C = A * scalar + B.
    /// Essential for gradient descent updates: W = grad * (-lr) + W.
    /// Dispatches to the persistent GPU ring buffer when available for large vectors (&gt;= 65536 under Auto)
    /// or explicit <see cref="GpuTarget.Nvidia"/>, otherwise executes hardware SIMD.
    /// Throws <see cref="NotSupportedException"/> for unsupported targets (<see cref="GpuTarget.Amd"/>, <see cref="GpuTarget.DualGpu"/>).
    /// </summary>
    public static void VectorFma(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> destination,
        float scalar,
        GpuTarget target = GpuTarget.Auto)
    {
        if (target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{target} is not supported by GpuMlAccelerator.VectorFma. Use Auto, Nvidia, or Cpu.");

        int count = a.Length;
        if (b.Length != count || destination.Length < count)
            throw new ArgumentException("Vector dimension mismatch.");

        if (count == 0) return;

        bool useGpu = target switch
        {
            GpuTarget.Cpu => false,
            GpuTarget.Nvidia => true,
            GpuTarget.Auto => count >= RingBufferAutoThreshold && IsNvidiaAvailable,
            _ => throw new NotSupportedException($"GpuTarget.{target} is not supported.")
        };

        if (useGpu && ExecuteRingBufferVectorOp(TaskOpCode.VectorFma, a, b, destination, scalar))
            return;

        VectorFmaSimd(a, b, destination, scalar);
    }

    private static bool ExecuteRingBufferVectorOp(
        TaskOpCode opCode,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> destination,
        float scalar)
    {
        if (!EnsureNvidiaInitialized()) return false;

        int count = a.Length;
        nuint bytes = checked((nuint)(uint)count * sizeof(float));

        lock (s_ringBufferLock)
        {
            try
            {
                lock (s_initLock)
                {
                    s_nvidiaEngine ??= new NvidiaSassEngine();
                    if (!s_nvidiaEngine.IsInitialized)
                        s_nvidiaEngine.Initialize();
                }

                if (CuDriver.CtxSetCurrent(s_nvidiaEngine.ContextHandle) != 0)
                    return false;

                if (!EnsureVectorBuffers(bytes))
                    return false;

                var ring = GetRingBuffer();
                if (ring == null) return false;

                var spanA = new Span<float>((void*)s_hVecA, count);
                var spanB = new Span<float>((void*)s_hVecB, count);
                a.CopyTo(spanA);
                b.CopyTo(spanB);

                ring.SubmitAndWait(opCode, (uint)count, (ulong)s_dVecA, (ulong)s_dVecB, (ulong)s_dVecC, scalar);

                var spanC = new ReadOnlySpan<float>((void*)s_hVecC, count);
                spanC.CopyTo(destination);
                return true;
            }
            catch
            {
                return false;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void VectorAddSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest)
    {
        int length = a.Length;
        int i = 0;
        ref float rA = ref MemoryMarshal.GetReference(a);
        ref float rB = ref MemoryMarshal.GetReference(b);
        ref float rD = ref MemoryMarshal.GetReference(dest);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector512.LoadUnsafe(ref rA, (nuint)i);
                var vb = Vector512.LoadUnsafe(ref rB, (nuint)i);
                (va + vb).StoreUnsafe(ref rD, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector256.LoadUnsafe(ref rA, (nuint)i);
                var vb = Vector256.LoadUnsafe(ref rB, (nuint)i);
                (va + vb).StoreUnsafe(ref rD, (nuint)i);
                i += step;
            }
        }

        for (; i < length; i++)
        {
            dest[i] = a[i] + b[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void VectorFmaSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> dest, float scalar)
    {
        int length = a.Length;
        int i = 0;
        ref float rA = ref MemoryMarshal.GetReference(a);
        ref float rB = ref MemoryMarshal.GetReference(b);
        ref float rD = ref MemoryMarshal.GetReference(dest);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var vScalar = Vector512.Create(scalar);
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector512.LoadUnsafe(ref rA, (nuint)i);
                var vb = Vector512.LoadUnsafe(ref rB, (nuint)i);
                Vector512.FusedMultiplyAdd(va, vScalar, vb).StoreUnsafe(ref rD, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var vScalar = Vector256.Create(scalar);
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector256.LoadUnsafe(ref rA, (nuint)i);
                var vb = Vector256.LoadUnsafe(ref rB, (nuint)i);
                Vector256.FusedMultiplyAdd(va, vScalar, vb).StoreUnsafe(ref rD, (nuint)i);
                i += step;
            }
        }

        for (; i < length; i++)
        {
            dest[i] = a[i] * scalar + b[i];
        }
    }

    #endregion
}

/// <summary>
/// Thread-safe, lock-free per-stream device execution context for GPU ML offload.
/// Each context owns its own dedicated CUDA stream and pre-allocated device memory slab.
/// </summary>
public sealed class GpuMlStreamContext : IDisposable
{
    public IntPtr Stream { get; }
    public IntPtr DeviceSamples { get; private set; }
    public IntPtr DeviceCentroids { get; private set; }
    public IntPtr DeviceAssignments { get; private set; }

    public nuint CapSamples { get; private set; }
    public nuint CapCentroids { get; private set; }
    public nuint CapAssignments { get; private set; }

    private bool _disposed;

    public GpuMlStreamContext()
    {
        if (GpuMlAccelerator.IsNvidiaAvailable && GpuMlAccelerator.ContextHandle != IntPtr.Zero)
        {
            if (CuDriver.CtxSetCurrent(GpuMlAccelerator.ContextHandle) == 0 &&
                CuDriver.StreamCreate(out IntPtr stream, CuDriver.CU_STREAM_NON_BLOCKING) == 0)
            {
                Stream = stream;
            }
        }
    }

    public void EnsureCapacity(nuint bytesSamples, nuint bytesCentroids, nuint bytesAssignments)
    {
        if (!GpuMlAccelerator.IsNvidiaAvailable || GpuMlAccelerator.ContextHandle == IntPtr.Zero)
            return;

        if (CuDriver.CtxSetCurrent(GpuMlAccelerator.ContextHandle) != 0)
            return;

        if (bytesSamples > CapSamples)
        {
            if (DeviceSamples != IntPtr.Zero)
            {
                CuDriver.MemFree(DeviceSamples);
                DeviceSamples = IntPtr.Zero;
                CapSamples = 0;
            }
            if (CuDriver.MemAlloc(out IntPtr dS, bytesSamples) == 0)
            {
                DeviceSamples = dS;
                CapSamples = bytesSamples;
            }
        }

        if (bytesCentroids > CapCentroids)
        {
            if (DeviceCentroids != IntPtr.Zero)
            {
                CuDriver.MemFree(DeviceCentroids);
                DeviceCentroids = IntPtr.Zero;
                CapCentroids = 0;
            }
            if (CuDriver.MemAlloc(out IntPtr dC, bytesCentroids) == 0)
            {
                DeviceCentroids = dC;
                CapCentroids = bytesCentroids;
            }
        }

        if (bytesAssignments > CapAssignments)
        {
            if (DeviceAssignments != IntPtr.Zero)
            {
                CuDriver.MemFree(DeviceAssignments);
                DeviceAssignments = IntPtr.Zero;
                CapAssignments = 0;
            }
            if (CuDriver.MemAlloc(out IntPtr dA, bytesAssignments) == 0)
            {
                DeviceAssignments = dA;
                CapAssignments = bytesAssignments;
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (GpuMlAccelerator.IsNvidiaAvailable && GpuMlAccelerator.ContextHandle != IntPtr.Zero &&
                CuDriver.CtxSetCurrent(GpuMlAccelerator.ContextHandle) == 0)
            {
                if (DeviceSamples != IntPtr.Zero) { CuDriver.MemFree(DeviceSamples); DeviceSamples = IntPtr.Zero; }
                if (DeviceCentroids != IntPtr.Zero) { CuDriver.MemFree(DeviceCentroids); DeviceCentroids = IntPtr.Zero; }
                if (DeviceAssignments != IntPtr.Zero) { CuDriver.MemFree(DeviceAssignments); DeviceAssignments = IntPtr.Zero; }
                if (Stream != IntPtr.Zero) CuDriver.StreamDestroy(Stream);
            }
        }
    }
}
