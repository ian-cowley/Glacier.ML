using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Core;

namespace Glacier.ML.Decomposition;

/// <summary>
/// Principal Component Analysis (PCA) for .NET 10 using double-precision covariance accumulation
/// and a Symmetric Cyclic Jacobi eigensolver on multi-core SIMD CPU (<see cref="GpuTarget.Auto"/> and <see cref="GpuTarget.Cpu"/>).
/// Equivalent to Scikit-Learn sklearn.decomposition.PCA.
/// </summary>
public sealed class FastPCA : IUnsupervisedEstimator<FastPCA>
{
    private readonly int _nComponents;
    private readonly GpuTarget _target;
    private int _fittedComponents;
    private float[] _components = Array.Empty<float>(); // [k, nFeatures]
    private float[] _mean = Array.Empty<float>();       // [nFeatures]
    private float[] _explainedVariance = Array.Empty<float>();
    private float[] _explainedVarianceRatio = Array.Empty<float>();
    private int _nFeatures;

    public int NComponents => _fittedComponents > 0 ? _fittedComponents : _nComponents;
    public GpuTarget Target => _target;
    public float[] Components => _components;
    public float[] Mean => _mean;
    public float[] ExplainedVariance => _explainedVariance;
    public float[] ExplainedVarianceRatio => _explainedVarianceRatio;
    public int NFeatures => _nFeatures;

    public FastPCA(int nComponents = 2, GpuTarget target = GpuTarget.Auto)
    {
        if (nComponents <= 0) throw new ArgumentOutOfRangeException(nameof(nComponents));
        _nComponents = nComponents;
        _target = target;
    }

    FastPCA IUnsupervisedEstimator<FastPCA>.Fit(FeatureMatrix features)
    {
        Fit(features);
        return this;
    }

    private void ValidateTarget()
    {
        if (_target is GpuTarget.Nvidia or GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{_target} is not supported by FastPCA (uses CPU SIMD + double-precision Jacobi). Use Auto or Cpu.");
    }

    /// <summary>
    /// Fits the PCA model on the feature matrix X.
    /// Computes column means and the double-precision covariance matrix without LOH centering copies,
    /// then extracts sorted orthonormal principal components via Symmetric Cyclic Jacobi eigendecomposition.
    /// </summary>
    public void Fit(FeatureMatrix features)
    {
        ArgumentNullException.ThrowIfNull(features);
        ValidateTarget();

        int rows = features.Rows;
        int cols = features.Columns;
        if (rows <= 0)
            throw new ArgumentException("FeatureMatrix must contain at least one row.", nameof(features));

        _nFeatures = cols;
        int k = Math.Min(_nComponents, cols);
        _fittedComponents = k;

        double[] meanD = new double[cols];
        _mean = new float[cols];
        double[] cov = new double[cols * cols];
        double factor = 1.0 / Math.Max(1, rows - 1);

        if (features.IsColumnar)
        {
            ReadOnlyMemory<float>[] colMemories = new ReadOnlyMemory<float>[cols];
            for (int c = 0; c < cols; c++)
            {
                colMemories[c] = features.GetColumnMemory(c);
                double sum = SumDoubleSimd(colMemories[c].Span);
                meanD[c] = sum / rows;
                _mean[c] = (float)meanD[c];
            }

            Parallel.For(0, cols, i =>
            {
                ReadOnlySpan<float> colI = colMemories[i].Span;
                double meanI = meanD[i];

                for (int j = i; j < cols; j++)
                {
                    ReadOnlySpan<float> colJ = colMemories[j].Span;
                    double dot = CenteredDotProductDouble(colI, meanI, colJ, meanD[j]);
                    double covVal = dot * factor;
                    cov[i * cols + j] = covVal;
                    cov[j * cols + i] = covVal;
                }
            });
        }
        else
        {
            ReadOnlySpan<float> data = features.Span;
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> row = data.Slice((int)((long)r * cols), cols);
                for (int c = 0; c < cols; c++)
                {
                    meanD[c] += row[c];
                }
            }

            for (int c = 0; c < cols; c++)
            {
                meanD[c] /= rows;
                _mean[c] = (float)meanD[c];
            }

            int numThreads = Math.Min(Environment.ProcessorCount, Math.Max(1, rows / 256));
            if (numThreads > 1)
            {
                double[][] threadCovs = new double[numThreads][];
                for (int t = 0; t < numThreads; t++) threadCovs[t] = new double[cols * cols];

                unsafe
                {
                    fixed (float* pData = data)
                    {
                        nint dataAddr = (nint)pData;
                        Parallel.For(0, numThreads, t =>
                        {
                            float* pRaw = (float*)dataAddr;
                            double[] localCov = threadCovs[t];
                            int startRow = (int)((long)t * rows / numThreads);
                            int endRow = (t == numThreads - 1) ? rows : (int)((long)(t + 1) * rows / numThreads);

                            double[]? rented = cols > 256 ? ArrayPool<double>.Shared.Rent(cols) : null;
                            try
                            {
                                Span<double> centeredRow = cols <= 256 ? stackalloc double[cols] : rented.AsSpan(0, cols);
                                for (int r = startRow; r < endRow; r++)
                                {
                                    float* pRow = pRaw + ((long)r * cols);
                                    for (int c = 0; c < cols; c++)
                                    {
                                        centeredRow[c] = pRow[c] - meanD[c];
                                    }

                                    for (int i = 0; i < cols; i++)
                                    {
                                        double ci = centeredRow[i];
                                        int iOffset = i * cols;
                                        for (int j = i; j < cols; j++)
                                        {
                                            localCov[iOffset + j] += ci * centeredRow[j];
                                        }
                                    }
                                }
                            }
                            finally
                            {
                                if (rented != null) ArrayPool<double>.Shared.Return(rented);
                            }
                        });
                    }
                }

                for (int t = 0; t < numThreads; t++)
                {
                    double[] localCov = threadCovs[t];
                    for (int i = 0; i < cols; i++)
                    {
                        int iOffset = i * cols;
                        for (int j = i; j < cols; j++)
                        {
                            cov[iOffset + j] += localCov[iOffset + j];
                        }
                    }
                }
            }
            else
            {
                double[]? rented = cols > 256 ? ArrayPool<double>.Shared.Rent(cols) : null;
                try
                {
                    Span<double> centeredRow = cols <= 256 ? stackalloc double[cols] : rented.AsSpan(0, cols);
                    for (int r = 0; r < rows; r++)
                    {
                        ReadOnlySpan<float> row = data.Slice((int)((long)r * cols), cols);
                        for (int c = 0; c < cols; c++)
                        {
                            centeredRow[c] = row[c] - meanD[c];
                        }

                        for (int i = 0; i < cols; i++)
                        {
                            double ci = centeredRow[i];
                            int iOffset = i * cols;
                            for (int j = i; j < cols; j++)
                            {
                                cov[iOffset + j] += ci * centeredRow[j];
                            }
                        }
                    }
                }
                finally
                {
                    if (rented != null) ArrayPool<double>.Shared.Return(rented);
                }
            }

            for (int i = 0; i < cols; i++)
            {
                int iOffset = i * cols;
                for (int j = i; j < cols; j++)
                {
                    double val = cov[iOffset + j] * factor;
                    cov[iOffset + j] = val;
                    cov[j * cols + i] = val;
                }
            }
        }

        SymmetricEigenSolver.SolveDescending(cov, cols, out double[] eigVals, out double[] eigVecs);

        double totalVariance = 0.0;
        for (int i = 0; i < cols; i++)
        {
            if (eigVals[i] > 0.0)
                totalVariance += eigVals[i];
        }

        _components = new float[k * cols];
        _explainedVariance = new float[k];
        _explainedVarianceRatio = new float[k];

        for (int c = 0; c < k; c++)
        {
            double ev = Math.Max(0.0, eigVals[c]);
            _explainedVariance[c] = (float)ev;
            _explainedVarianceRatio[c] = totalVariance > 0.0 ? (float)(ev / totalVariance) : 0f;

            int offset = c * cols;
            for (int d = 0; d < cols; d++)
            {
                _components[offset + d] = (float)eigVecs[offset + d];
            }
        }
    }

    /// <summary>
    /// Projects feature matrix X onto the principal components: Y = (X - mean) * W^T.
    /// Executes directly via zero-copy SIMD dot products without intermediate matrix allocation.
    /// </summary>
    public void Transform(FeatureMatrix features, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(features);
        ValidateTarget();
        if (_nFeatures <= 0 || _components.Length == 0 || _explainedVariance.Length == 0)
            throw new InvalidOperationException("FastPCA must be fit before calling Transform.");

        int rows = features.Rows;
        int cols = features.Columns;
        int k = _explainedVariance.Length;

        if (cols != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {cols}.", nameof(features));
        if ((long)rows * k > destination.Length)
            throw new ArgumentException($"Destination span too small. Required {(long)rows * k}, got {destination.Length}.", nameof(destination));

        if (rows == 0) return;

        if (features.IsColumnar)
        {
            ReadOnlyMemory<float>[] colMemories = new ReadOnlyMemory<float>[cols];
            for (int d = 0; d < cols; d++)
            {
                colMemories[d] = features.GetColumnMemory(d);
            }

            unsafe
            {
                fixed (float* pDest = destination)
                fixed (float* pComp = _components)
                fixed (float* pMean = _mean)
                {
                    nint destAddr = (nint)pDest;
                    nint compAddr = (nint)pComp;
                    nint meanAddr = (nint)pMean;

                    Parallel.For(0, rows, r =>
                    {
                        float* pOut = (float*)destAddr + ((long)r * k);
                        float* pC = (float*)compAddr;
                        float* pM = (float*)meanAddr;

                        float[]? rented = cols > 256 ? ArrayPool<float>.Shared.Rent(cols) : null;
                        try
                        {
                            Span<float> centered = cols <= 256 ? stackalloc float[cols] : rented.AsSpan(0, cols);
                            for (int d = 0; d < cols; d++)
                            {
                                centered[d] = colMemories[d].Span[r] - pM[d];
                            }

                            for (int c = 0; c < k; c++)
                            {
                                pOut[c] = DotProductSimd(centered, new ReadOnlySpan<float>(pC + (long)c * cols, cols));
                            }
                        }
                        finally
                        {
                            if (rented != null) ArrayPool<float>.Shared.Return(rented);
                        }
                    });
                }
            }
            return;
        }

        unsafe
        {
            fixed (float* pData = features.Span)
            fixed (float* pDest = destination)
            fixed (float* pComp = _components)
            fixed (float* pMean = _mean)
            {
                nint dataAddr = (nint)pData;
                nint destAddr = (nint)pDest;
                nint compAddr = (nint)pComp;
                nint meanAddr = (nint)pMean;

                Parallel.For(0, rows, r =>
                {
                    var row = new ReadOnlySpan<float>((float*)dataAddr + ((long)r * cols), cols);
                    var mean = new ReadOnlySpan<float>((float*)meanAddr, cols);
                    float* pOut = (float*)destAddr + ((long)r * k);
                    float* pC = (float*)compAddr;

                    for (int c = 0; c < k; c++)
                    {
                        var comp = new ReadOnlySpan<float>(pC + ((long)c * cols), cols);
                        pOut[c] = CenteredRowDotProductSimd(row, mean, comp);
                    }
                });
            }
        }
    }

    /// <summary>
    /// Reconstructs original data from the reduced dimension space: X_reconstructed = Y * W + mean.
    /// </summary>
    public void InverseTransform(ReadOnlySpan<float> reduced, Span<float> destination, int rows)
    {
        if (_nFeatures <= 0 || _components.Length == 0 || _explainedVariance.Length == 0)
            throw new InvalidOperationException("FastPCA must be fit before calling InverseTransform.");
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Rows must be non-negative.");
        if (rows == 0) return;

        int cols = _nFeatures;
        int k = _explainedVariance.Length;

        if ((long)rows * k > reduced.Length)
            throw new ArgumentException("Reduced span too small.", nameof(reduced));
        if ((long)rows * cols > destination.Length)
            throw new ArgumentException("Destination span too small.", nameof(destination));

        unsafe
        {
            fixed (float* pRed = reduced)
            fixed (float* pDest = destination)
            fixed (float* pComp = _components)
            fixed (float* pMean = _mean)
            {
                for (int r = 0; r < rows; r++)
                {
                    long rOffset = (long)r * cols;
                    long redOffset = (long)r * k;

                    for (int d = 0; d < cols; d++)
                    {
                        float sum = pMean[d];
                        for (int c = 0; c < k; c++)
                        {
                            sum += pRed[redOffset + c] * pComp[(long)c * cols + d];
                        }
                        pDest[rOffset + d] = sum;
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double SumDoubleSimd(ReadOnlySpan<float> col)
    {
        int n = col.Length;
        int vecSize = Vector<float>.Count;
        var vAccLo = Vector<double>.Zero;
        var vAccHi = Vector<double>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            var vf = new Vector<float>(col.Slice(i, vecSize));
            Vector.Widen(vf, out var lo, out var hi);
            vAccLo += lo;
            vAccHi += hi;
        }

        double sum = Vector.Sum(vAccLo) + Vector.Sum(vAccHi);
        for (; i < n; i++)
        {
            sum += col[i];
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double CenteredDotProductDouble(
        ReadOnlySpan<float> colA, double meanA,
        ReadOnlySpan<float> colB, double meanB)
    {
        int n = colA.Length;
        int vecSize = Vector<float>.Count;
        var vMeanA = new Vector<double>(meanA);
        var vMeanB = new Vector<double>(meanB);
        var vAccLo = Vector<double>.Zero;
        var vAccHi = Vector<double>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            Vector.Widen(new Vector<float>(colA.Slice(i, vecSize)), out var aLo, out var aHi);
            Vector.Widen(new Vector<float>(colB.Slice(i, vecSize)), out var bLo, out var bHi);
            vAccLo += (aLo - vMeanA) * (bLo - vMeanB);
            vAccHi += (aHi - vMeanA) * (bHi - vMeanB);
        }

        double sum = Vector.Sum(vAccLo) + Vector.Sum(vAccHi);
        for (; i < n; i++)
        {
            sum += (colA[i] - meanA) * (colB[i] - meanB);
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float CenteredRowDotProductSimd(
        ReadOnlySpan<float> row,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> comp)
    {
        int n = row.Length;
        int vecSize = Vector<float>.Count;
        var vAcc = Vector<float>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            var vr = new Vector<float>(row.Slice(i, vecSize)) - new Vector<float>(mean.Slice(i, vecSize));
            var vc = new Vector<float>(comp.Slice(i, vecSize));
            vAcc += vr * vc;
        }

        float sum = Vector.Sum(vAcc);
        for (; i < n; i++)
        {
            sum += (row[i] - mean[i]) * comp[i];
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotProductSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = a.Length;
        int vecSize = Vector<float>.Count;
        var vAcc = Vector<float>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            vAcc += new Vector<float>(a.Slice(i, vecSize)) * new Vector<float>(b.Slice(i, vecSize));
        }

        float sum = Vector.Sum(vAcc);
        for (; i < n; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
