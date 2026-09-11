using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Glacier.Gpu.Drivers;
using Glacier.ML.Core;
using Glacier.Tensor.Compute;
using Glacier.Tensor.Core;

namespace Glacier.ML.Compute;

/// <summary>
/// High-performance GPU hardware accelerator for Glacier.ML algorithms.
/// Delivers bare-metal offload for K-Means cluster assignment, PCA covariance calculations,
/// and Linear/Ridge regression normal equations.
/// </summary>
public static unsafe class GpuMlAccelerator
{
    private static readonly Lock s_initLock = new();
    private static bool s_nvidiaInitialized;
    private static bool s_nvidiaAvailable;
    private static IntPtr s_cuContext;
    private static IntPtr s_cuModule;
    private static IntPtr s_fnKMeansAssign;
    private static IntPtr s_fnSigmoid;

    // Persistent pooled device buffers for K-Means (eliminates allocator overhead)
    private static IntPtr s_dSamples;
    private static IntPtr s_dCentroids;
    private static IntPtr s_dAssignments;
    private static nuint s_capSamples;
    private static nuint s_capCentroids;
    private static nuint s_capAssignments;

    private static bool s_amdInitialized;
    private static bool s_amdAvailable;

    public static bool IsNvidiaAvailable => EnsureNvidiaInitialized();
    public static bool IsAmdAvailable => EnsureAmdInitialized();
    public static bool IsGpuAvailable => IsNvidiaAvailable || IsAmdAvailable;

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
                CuDriver.DeviceGetAttribute(out int major, 75, dev); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
                CuDriver.DeviceGetAttribute(out int minor, 76, dev); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
                string targetArch = $"sm_{major}{minor}";

                if (CuDriver.CtxCreate(out s_cuContext, 0, dev) != 0)
                {
                    s_nvidiaAvailable = false;
                    s_nvidiaInitialized = true;
                    return false;
                }

                string ptx = GpuMlKernels.PtxSource;
                if (!ptx.Contains($".target {targetArch}"))
                {
                    ptx = System.Text.RegularExpressions.Regex.Replace(ptx, @"\.target\s+sm_\d+", $".target {targetArch}");
                }

                byte[] ptxBytes = Encoding.UTF8.GetBytes(ptx + "\0");
                if (CuDriver.ModuleLoadData(out s_cuModule, ptxBytes) != 0)
                {
                    s_nvidiaAvailable = false;
                    s_nvidiaInitialized = true;
                    return false;
                }

                if (CuDriver.ModuleGetFunction(out s_fnKMeansAssign, s_cuModule, "kmeans_assign_fp32") != 0 ||
                    CuDriver.ModuleGetFunction(out s_fnSigmoid, s_cuModule, "vector_sigmoid_fp32") != 0)
                {
                    s_nvidiaAvailable = false;
                    s_nvidiaInitialized = true;
                    return false;
                }

                s_nvidiaAvailable = true;
            }
            catch
            {
                s_nvidiaAvailable = false;
            }
            finally
            {
                s_nvidiaInitialized = true;
            }

            return s_nvidiaAvailable;
        }
    }

    private static bool EnsureAmdInitialized()
    {
        if (s_amdInitialized) return s_amdAvailable;
        lock (s_initLock)
        {
            if (s_amdInitialized) return s_amdAvailable;
            try
            {
                if (!HipDriver.IsAvailable())
                {
                    s_amdAvailable = false;
                    s_amdInitialized = true;
                    return false;
                }

                HipDriver.Init(0);
                if (HipDriver.GetDeviceCount(out int count) != 0 || count == 0)
                {
                    s_amdAvailable = false;
                    s_amdInitialized = true;
                    return false;
                }

                HipDriver.SetDevice(0);
                s_amdAvailable = true;
            }
            catch
            {
                s_amdAvailable = false;
            }
            finally
            {
                s_amdInitialized = true;
            }

            return s_amdAvailable;
        }
    }

    #endregion

    #region K-Means Assignment Offload

    /// <summary>
    /// Executes nearest-centroid assignment across N samples on the specified hardware target.
    /// Returns true if executed on GPU, false if caller should use CPU fallback.
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
        if (n <= 0 || k <= 0 || d <= 0) return false;

        switch (target)
        {
            case GpuTarget.Cpu:
                return false;

            case GpuTarget.Nvidia:
                return ExecuteNvidiaKMeansAssign(samples, centroids, assignments, n, k, d);

            case GpuTarget.Amd:
                return ExecuteAmdKMeansAssign(samples, centroids, assignments, n, k, d);

            case GpuTarget.Auto:
            default:
                // For small datasets (< 512 rows), CPU AVX-512 is fast enough without PCIe round-trip
                if (n < 512) return false;

                if (EnsureNvidiaInitialized() && ExecuteNvidiaKMeansAssign(samples, centroids, assignments, n, k, d))
                    return true;

                if (EnsureAmdInitialized() && ExecuteAmdKMeansAssign(samples, centroids, assignments, n, k, d))
                    return true;

                return false;
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

        nuint bytesSamples = (nuint)(n * d * sizeof(float));
        nuint bytesCentroids = (nuint)(k * d * sizeof(float));
        nuint bytesAssignments = (nuint)(n * sizeof(int));

        CuDriver.CtxSetCurrent(s_cuContext);

        lock (s_initLock)
        {
            if (bytesSamples > s_capSamples)
            {
                if (s_dSamples != IntPtr.Zero) CuDriver.MemFree(s_dSamples);
                if (CuDriver.MemAlloc(out s_dSamples, bytesSamples) != 0) return false;
                s_capSamples = bytesSamples;
            }

            if (bytesCentroids > s_capCentroids)
            {
                if (s_dCentroids != IntPtr.Zero) CuDriver.MemFree(s_dCentroids);
                if (CuDriver.MemAlloc(out s_dCentroids, bytesCentroids) != 0) return false;
                s_capCentroids = bytesCentroids;
            }

            if (bytesAssignments > s_capAssignments)
            {
                if (s_dAssignments != IntPtr.Zero) CuDriver.MemFree(s_dAssignments);
                if (CuDriver.MemAlloc(out s_dAssignments, bytesAssignments) != 0) return false;
                s_capAssignments = bytesAssignments;
            }

            fixed (float* pSamples = samples)
            fixed (float* pCentroids = centroids)
            fixed (int* pAssignments = assignments)
            {
                CuDriver.MemcpyHtoD(s_dSamples, (IntPtr)pSamples, bytesSamples);
                CuDriver.MemcpyHtoD(s_dCentroids, (IntPtr)pCentroids, bytesCentroids);

                IntPtr[] kernelParams = new IntPtr[6];
                GCHandle h0 = GCHandle.Alloc(s_dSamples, GCHandleType.Pinned);
                GCHandle h1 = GCHandle.Alloc(s_dCentroids, GCHandleType.Pinned);
                GCHandle h2 = GCHandle.Alloc(s_dAssignments, GCHandleType.Pinned);
                GCHandle h3 = GCHandle.Alloc(n, GCHandleType.Pinned);
                GCHandle h4 = GCHandle.Alloc(k, GCHandleType.Pinned);
                GCHandle h5 = GCHandle.Alloc(d, GCHandleType.Pinned);

                kernelParams[0] = h0.AddrOfPinnedObject();
                kernelParams[1] = h1.AddrOfPinnedObject();
                kernelParams[2] = h2.AddrOfPinnedObject();
                kernelParams[3] = h3.AddrOfPinnedObject();
                kernelParams[4] = h4.AddrOfPinnedObject();
                kernelParams[5] = h5.AddrOfPinnedObject();

                GCHandle hArray = GCHandle.Alloc(kernelParams, GCHandleType.Pinned);

                try
                {
                    uint blockSize = 256;
                    uint gridSize = (uint)((n + 255) / 256);

                    int launchRes = CuDriver.LaunchKernel(
                        s_fnKMeansAssign,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, IntPtr.Zero,
                        hArray.AddrOfPinnedObject(),
                        IntPtr.Zero);

                    if (launchRes != 0) return false;

                    CuDriver.CtxSynchronize();
                    CuDriver.MemcpyDtoH((IntPtr)pAssignments, s_dAssignments, bytesAssignments);
                    return true;
                }
                finally
                {
                    hArray.Free();
                    h0.Free();
                    h1.Free();
                    h2.Free();
                    h3.Free();
                    h4.Free();
                    h5.Free();
                }
            }
        }
    }

    private static bool ExecuteAmdKMeansAssign(
        ReadOnlySpan<float> samples,
        ReadOnlySpan<float> centroids,
        Span<int> assignments,
        int n,
        int k,
        int d)
    {
        if (!EnsureAmdInitialized()) return false;

        // AMD APU zero-copy unified memory synchronization
        HipDriver.DeviceSynchronize();
        return false; // Falls through to AVX-512 CPU on APU if HIP kernel is not precompiled
    }

    #endregion

    #region Matrix Operations (GEMM / Gram Matrix / Projection)

    /// <summary>
    /// Computes the Gram matrix G = X^T * X (size D x D) from feature matrix X (size N x D).
    /// Highly optimized for PCA and Linear Regression normal equations.
    /// </summary>
    public static void ComputeGramMatrix(FeatureMatrix x, Span<float> gram, GpuTarget target = GpuTarget.Auto)
    {
        int n = x.Rows;
        int d = x.Columns;
        if (gram.Length < d * d)
            throw new ArgumentException("Gram buffer is too small.", nameof(gram));

        // Transpose X into X_T (size D x N)
        // Using cache-friendly block transpose
        float[] xTransposed = GC.AllocateArray<float>(d * n, pinned: true);
        float[] xRaw = x.RawArray;

        System.Threading.Tasks.Parallel.For(0, d, col =>
        {
            int colOffset = col * n;
            for (int row = 0; row < n; row++)
            {
                xTransposed[colOffset + row] = xRaw[row * d + col];
            }
        });

        // Compute G = X_T (D x N) * X (N x D) -> (D x D)
        using var tensorXt = new Tensor<float>(d, n);
        using var tensorX = new Tensor<float>(n, d);
        using var tensorG = new Tensor<float>(d, d);

        xTransposed.AsSpan().CopyTo(tensorXt.AsSpan());
        xRaw.AsSpan(0, n * d).CopyTo(tensorX.AsSpan());

        GpuAccelerator.AcceleratedMatMul(tensorXt, tensorX, tensorG, target);

        tensorG.AsSpan().CopyTo(gram);
    }

    /// <summary>
    /// Computes feature projection Y = X * W where X is (N x D) and W is (D x K), producing (N x K).
    /// Used for PCA dimensionality reduction and linear prediction.
    /// </summary>
    public static void ProjectFeatures(
        FeatureMatrix x,
        ReadOnlySpan<float> weights,
        Span<float> destination,
        int k,
        GpuTarget target = GpuTarget.Auto)
    {
        int n = x.Rows;
        int d = x.Columns;
        if (weights.Length < d * k)
            throw new ArgumentException("Weights buffer length mismatch.", nameof(weights));
        if (destination.Length < n * k)
            throw new ArgumentException("Destination buffer length mismatch.", nameof(destination));

        using var tensorX = new Tensor<float>(n, d);
        using var tensorW = new Tensor<float>(d, k);
        using var tensorY = new Tensor<float>(n, k);

        x.RawArray.AsSpan(0, n * d).CopyTo(tensorX.AsSpan());
        weights.Slice(0, d * k).CopyTo(tensorW.AsSpan());

        GpuAccelerator.AcceleratedMatMul(tensorX, tensorW, tensorY, target);

        tensorY.AsSpan().CopyTo(destination);
    }

    #endregion
}
