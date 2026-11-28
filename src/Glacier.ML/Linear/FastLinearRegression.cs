using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.ML.Core;

namespace Glacier.ML.Linear;

/// <summary>
/// Ordinary Least Squares (OLS) and Ridge Regression engine for .NET 10.
/// Accumulates normal equations in double precision and solves via Cholesky decomposition
/// with a double-precision Symmetric Jacobi pseudoinverse fallback for rank-deficient inputs,
/// running on multi-core SIMD CPU (<see cref="GpuTarget.Auto"/> and <see cref="GpuTarget.Cpu"/>).
/// Equivalent to Scikit-Learn LinearRegression and Ridge.
/// </summary>
public sealed class FastLinearRegression : IPredictor, IEstimator<FastLinearRegression>
{
    private readonly float _alpha;
    private readonly bool _fitIntercept;
    private readonly GpuTarget _target;
    private float[] _weights = Array.Empty<float>();
    private float _intercept;
    private int _nFeatures;

    public float[] Weights => _weights;
    public float Intercept => _intercept;
    public float Alpha => _alpha;
    public bool FitIntercept => _fitIntercept;
    public GpuTarget Target => _target;
    public int NFeatures => _nFeatures;

    public FastLinearRegression(float alpha = 0.0f, bool fitIntercept = true, GpuTarget target = GpuTarget.Auto)
    {
        if (!float.IsFinite(alpha) || alpha < 0f)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be finite and non-negative.");
        _alpha = alpha;
        _fitIntercept = fitIntercept;
        _target = target;
    }

    FastLinearRegression IEstimator<FastLinearRegression>.Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        Fit(features, targets);
        return this;
    }

    private void ValidateTarget()
    {
        if (_target is GpuTarget.Nvidia or GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{_target} is not supported by FastLinearRegression (uses CPU SIMD + double-precision normal equations). Use Auto or Cpu.");
    }

    /// <summary>
    /// Fits the linear model using double-precision normal equations (no hidden regularization when alpha == 0).
    /// </summary>
    public unsafe void Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        ArgumentNullException.ThrowIfNull(features);
        ValidateTarget();

        int rows = features.Rows;
        int cols = features.Columns;
        if (rows <= 0)
            throw new ArgumentException("FeatureMatrix must contain at least one row.", nameof(features));
        if (targets.Length < rows)
            throw new ArgumentException("Targets length does not match features row count.", nameof(targets));

        _nFeatures = cols;
        _weights = new float[cols];

        double[] gram = new double[cols * cols];
        double[] rhs = new double[cols];
        double[] meanX = _fitIntercept ? new double[cols] : Array.Empty<double>();
        double meanY = 0.0;

        if (_fitIntercept)
        {
            double sumY = 0.0;
            for (int r = 0; r < rows; r++) sumY += targets[r];
            meanY = sumY / rows;
        }

        if (features.IsColumnar)
        {
            ReadOnlyMemory<float>[] colMemories = new ReadOnlyMemory<float>[cols];
            for (int c = 0; c < cols; c++)
            {
                colMemories[c] = features.GetColumnMemory(c);
                if (_fitIntercept)
                {
                    meanX[c] = SumDouble(colMemories[c].Span) / rows;
                }
            }

            fixed (float* pTargets = targets)
            {
                nint targetsAddr = (nint)pTargets;
                Parallel.For(0, cols, i =>
                {
                    ReadOnlySpan<float> colI = colMemories[i].Span;
                    double mI = _fitIntercept ? meanX[i] : 0.0;

                    for (int j = i; j < cols; j++)
                    {
                        double dot = CenteredDotDouble(colI, mI, colMemories[j].Span, _fitIntercept ? meanX[j] : 0.0);
                        gram[i * cols + j] = dot;
                        gram[j * cols + i] = dot;
                    }

                    var ySpan = new ReadOnlySpan<float>((float*)targetsAddr, rows);
                    rhs[i] = CenteredDotDouble(colI, mI, ySpan, meanY);
                });
            }
        }
        else
        {
            ReadOnlySpan<float> data = features.Span;
            if (_fitIntercept)
            {
                for (int r = 0; r < rows; r++)
                {
                    ReadOnlySpan<float> row = data.Slice((int)((long)r * cols), cols);
                    for (int c = 0; c < cols; c++) meanX[c] += row[c];
                }
                for (int c = 0; c < cols; c++) meanX[c] /= rows;
            }

            int numThreads = Math.Min(Environment.ProcessorCount, Math.Max(1, rows / 256));
            if (numThreads > 1)
            {
                double[][] threadGrams = new double[numThreads][];
                double[][] threadRhs = new double[numThreads][];
                for (int t = 0; t < numThreads; t++)
                {
                    threadGrams[t] = new double[cols * cols];
                    threadRhs[t] = new double[cols];
                }

                fixed (float* pData = data)
                fixed (float* pTargets = targets)
                {
                    nint dataAddr = (nint)pData;
                    nint targetsAddr = (nint)pTargets;

                    Parallel.For(0, numThreads, t =>
                    {
                        float* pRaw = (float*)dataAddr;
                        float* pY = (float*)targetsAddr;
                        double[] localG = threadGrams[t];
                        double[] localR = threadRhs[t];
                        int startRow = (int)((long)t * rows / numThreads);
                        int endRow = (t == numThreads - 1) ? rows : (int)((long)(t + 1) * rows / numThreads);

                        double[]? rented = cols > 256 ? ArrayPool<double>.Shared.Rent(cols) : null;
                        try
                        {
                            Span<double> xRow = cols <= 256 ? stackalloc double[cols] : rented.AsSpan(0, cols);
                            for (int r = startRow; r < endRow; r++)
                            {
                                float* pRow = pRaw + ((long)r * cols);
                                double yVal = _fitIntercept ? (pY[r] - meanY) : pY[r];
                                for (int c = 0; c < cols; c++)
                                {
                                    xRow[c] = _fitIntercept ? (pRow[c] - meanX[c]) : pRow[c];
                                }

                                for (int i = 0; i < cols; i++)
                                {
                                    double xi = xRow[i];
                                    localR[i] += xi * yVal;
                                    int iOffset = i * cols;
                                    for (int j = i; j < cols; j++)
                                    {
                                        localG[iOffset + j] += xi * xRow[j];
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

                for (int t = 0; t < numThreads; t++)
                {
                    double[] localG = threadGrams[t];
                    double[] localR = threadRhs[t];
                    for (int i = 0; i < cols; i++)
                    {
                        rhs[i] += localR[i];
                        int iOffset = i * cols;
                        for (int j = i; j < cols; j++)
                        {
                            gram[iOffset + j] += localG[iOffset + j];
                        }
                    }
                }
            }
            else
            {
                double[]? rented = cols > 256 ? ArrayPool<double>.Shared.Rent(cols) : null;
                try
                {
                    Span<double> xRow = cols <= 256 ? stackalloc double[cols] : rented.AsSpan(0, cols);
                    for (int r = 0; r < rows; r++)
                    {
                        ReadOnlySpan<float> row = data.Slice((int)((long)r * cols), cols);
                        double yVal = _fitIntercept ? (targets[r] - meanY) : targets[r];
                        for (int c = 0; c < cols; c++)
                        {
                            xRow[c] = _fitIntercept ? (row[c] - meanX[c]) : row[c];
                        }

                        for (int i = 0; i < cols; i++)
                        {
                            double xi = xRow[i];
                            rhs[i] += xi * yVal;
                            int iOffset = i * cols;
                            for (int j = i; j < cols; j++)
                            {
                                gram[iOffset + j] += xi * xRow[j];
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
                for (int j = i + 1; j < cols; j++)
                {
                    gram[j * cols + i] = gram[i * cols + j];
                }
            }
        }

        // Only add Ridge regularization when alpha > 0 (never inject hidden bias when alpha == 0)
        if (_alpha > 0f)
        {
            double reg = _alpha;
            for (int i = 0; i < cols; i++)
            {
                gram[i * cols + i] += reg;
            }
        }

        double[] wDouble = new double[cols];
        SolveLinearSystemDouble(gram, rhs, wDouble, cols);

        for (int c = 0; c < cols; c++)
        {
            _weights[c] = (float)wDouble[c];
        }

        if (_fitIntercept)
        {
            double wDotMeanX = 0.0;
            for (int c = 0; c < cols; c++)
            {
                wDotMeanX += wDouble[c] * meanX[c];
            }
            _intercept = (float)(meanY - wDotMeanX);
        }
        else
        {
            _intercept = 0f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<float> predictions)
    {
        ArgumentNullException.ThrowIfNull(features);
        ValidateTarget();
        if (_weights.Length == 0 || _nFeatures <= 0)
            throw new InvalidOperationException("FastLinearRegression must be fit before calling Predict.");

        int rows = features.Rows;
        int cols = features.Columns;
        if (cols != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {cols}.", nameof(features));
        if (predictions.Length < rows)
            throw new ArgumentException("Predictions span too small.", nameof(predictions));

        if (rows == 0) return;

        if (features.IsColumnar)
        {
            predictions.Slice(0, rows).Fill(_intercept);
            for (int c = 0; c < cols; c++)
            {
                AccumulateScaledColumnSimd(features.GetColumn(c), _weights[c], predictions.Slice(0, rows));
            }
            return;
        }

        unsafe
        {
            fixed (float* pData = features.Span)
            fixed (float* pPred = predictions)
            fixed (float* pWeights = _weights)
            {
                float intercept = _intercept;
                if (rows >= 1024 && Environment.ProcessorCount > 1)
                {
                    nint dataAddr = (nint)pData;
                    nint predAddr = (nint)pPred;
                    nint weightsAddr = (nint)pWeights;
                    Parallel.For(0, rows, r =>
                    {
                        var row = new ReadOnlySpan<float>((float*)dataAddr + ((long)r * cols), cols);
                        var w = new ReadOnlySpan<float>((float*)weightsAddr, cols);
                        ((float*)predAddr)[r] = LinearKernels.DotProduct(row, w) + intercept;
                    });
                }
                else
                {
                    var w = new ReadOnlySpan<float>(pWeights, cols);
                    for (int r = 0; r < rows; r++)
                    {
                        var row = new ReadOnlySpan<float>(pData + ((long)r * cols), cols);
                        pPred[r] = LinearKernels.DotProduct(row, w) + intercept;
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictRow(ReadOnlySpan<float> row)
    {
        if (_weights.Length == 0 || _nFeatures <= 0)
            throw new InvalidOperationException("FastLinearRegression must be fit before calling PredictRow.");
        if (row.Length != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {row.Length}.", nameof(row));

        return LinearKernels.DotProduct(row, _weights) + _intercept;
    }

    /// <summary>
    /// Solves symmetric positive semi-definite linear system A * x = b in double precision
    /// using Cholesky decomposition, falling back to a Symmetric Jacobi pseudoinverse if rank-deficient.
    /// </summary>
    private static void SolveLinearSystemDouble(double[] a, double[] b, double[] x, int n)
    {
        double maxDiag = 0.0;
        for (int i = 0; i < n; i++)
        {
            double diag = Math.Abs(a[i * n + i]);
            if (diag > maxDiag) maxDiag = diag;
        }

        if (maxDiag <= 0.0)
        {
            Array.Clear(x);
            return;
        }

        double cholTol = 1e-14 * maxDiag;
        double[] l = new double[n * n];
        bool choleskySuccess = true;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < j; k++)
                {
                    sum += l[i * n + k] * l[j * n + k];
                }

                if (i == j)
                {
                    double val = a[i * n + i] - sum;
                    if (val <= cholTol)
                    {
                        choleskySuccess = false;
                        break;
                    }
                    l[i * n + j] = Math.Sqrt(val);
                }
                else
                {
                    l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
                }
            }
            if (!choleskySuccess) break;
        }

        if (choleskySuccess)
        {
            double[] y = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int k = 0; k < i; k++) sum += l[i * n + k] * y[k];
                y[i] = (b[i] - sum) / l[i * n + i];
            }

            for (int i = n - 1; i >= 0; i--)
            {
                double sum = 0.0;
                for (int k = i + 1; k < n; k++) sum += l[k * n + i] * x[k];
                x[i] = (y[i] - sum) / l[i * n + i];
            }
            return;
        }

        // Rank-deficient / near-singular fallback: Symmetric Jacobi minimum-norm pseudoinverse
        SymmetricEigenSolver.SolveDescending(a, n, out double[] eigVals, out double[] eigVecs);
        double maxEig = eigVals[0];
        Array.Clear(x);
        if (maxEig <= 0.0) return;

        double eigTol = 1e-12 * maxEig;
        for (int c = 0; c < n; c++)
        {
            double ev = eigVals[c];
            if (ev <= eigTol) break;

            int offset = c * n;
            double dot = 0.0;
            for (int d = 0; d < n; d++)
            {
                dot += eigVecs[offset + d] * b[d];
            }

            double scale = dot / ev;
            for (int d = 0; d < n; d++)
            {
                x[d] += scale * eigVecs[offset + d];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double SumDouble(ReadOnlySpan<float> col)
    {
        int n = col.Length;
        int vecSize = Vector<float>.Count;
        var vAccLo = Vector<double>.Zero;
        var vAccHi = Vector<double>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            Vector.Widen(new Vector<float>(col.Slice(i, vecSize)), out var lo, out var hi);
            vAccLo += lo;
            vAccHi += hi;
        }

        double sum = Vector.Sum(vAccLo) + Vector.Sum(vAccHi);
        for (; i < n; i++) sum += col[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double CenteredDotDouble(ReadOnlySpan<float> a, double meanA, ReadOnlySpan<float> b, double meanB)
    {
        int n = a.Length;
        int vecSize = Vector<float>.Count;
        var vMeanA = new Vector<double>(meanA);
        var vMeanB = new Vector<double>(meanB);
        var vAccLo = Vector<double>.Zero;
        var vAccHi = Vector<double>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            Vector.Widen(new Vector<float>(a.Slice(i, vecSize)), out var aLo, out var aHi);
            Vector.Widen(new Vector<float>(b.Slice(i, vecSize)), out var bLo, out var bHi);
            vAccLo += (aLo - vMeanA) * (bLo - vMeanB);
            vAccHi += (aHi - vMeanA) * (bHi - vMeanB);
        }

        double sum = Vector.Sum(vAccLo) + Vector.Sum(vAccHi);
        for (; i < n; i++)
        {
            sum += (a[i] - meanA) * (b[i] - meanB);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AccumulateScaledColumnSimd(ReadOnlySpan<float> col, float weight, Span<float> dst)
    {
        int length = col.Length;
        int i = 0;
        ref float rCol = ref MemoryMarshal.GetReference(col);
        ref float rDst = ref MemoryMarshal.GetReference(dst);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var vw = Vector512.Create(weight);
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vc = Vector512.LoadUnsafe(ref rCol, (nuint)i);
                var vd = Vector512.LoadUnsafe(ref rDst, (nuint)i);
                Vector512.FusedMultiplyAdd(vc, vw, vd).StoreUnsafe(ref rDst, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var vw = Vector256.Create(weight);
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vc = Vector256.LoadUnsafe(ref rCol, (nuint)i);
                var vd = Vector256.LoadUnsafe(ref rDst, (nuint)i);
                Vector256.FusedMultiplyAdd(vc, vw, vd).StoreUnsafe(ref rDst, (nuint)i);
                i += step;
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            var vw = Vector128.Create(weight);
            int step = Vector128<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vc = Vector128.LoadUnsafe(ref rCol, (nuint)i);
                var vd = Vector128.LoadUnsafe(ref rDst, (nuint)i);
                Vector128.FusedMultiplyAdd(vc, vw, vd).StoreUnsafe(ref rDst, (nuint)i);
                i += step;
            }
        }

        for (; i < length; i++)
        {
            dst[i] += col[i] * weight;
        }
    }
}
