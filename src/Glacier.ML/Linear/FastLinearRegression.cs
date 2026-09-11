using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Compute;
using Glacier.ML.Core;

namespace Glacier.ML.Linear;

/// <summary>
/// Hardware-accelerated Linear and Ridge Regression engine for .NET 10.
/// Solves normal equations (X^T * X + alpha * I) * w = X^T * y using GPU Gram matrix GEMM.
/// Equivalent to Scikit-Learn LinearRegression and Ridge.
/// </summary>
public sealed class FastLinearRegression : IPredictor
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
        if (alpha < 0f) throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be non-negative.");
        _alpha = alpha;
        _fitIntercept = fitIntercept;
        _target = target;
    }

    /// <summary>
    /// Fits the linear model using closed-form normal equations.
    /// Forms Gram matrix X^T * X on GPU and solves via Cholesky / LU decomposition.
    /// </summary>
    public void Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        int rows = features.Rows;
        int cols = features.Columns;
        _nFeatures = cols;

        if (targets.Length < rows)
            throw new ArgumentException("Targets length does not match features row count.", nameof(targets));

        _weights = new float[cols];

        if (_fitIntercept)
        {
            // Compute mean of targets and columns
            double sumY = 0;
            for (int r = 0; r < rows; r++) sumY += targets[r];
            float meanY = (float)(sumY / rows);

            float[] meanX = new float[cols];
            for (int c = 0; c < cols; c++)
            {
                double sumCol = 0;
                for (int r = 0; r < rows; r++) sumCol += features.GetValue(r, c);
                meanX[c] = (float)(sumCol / rows);
            }

            // Create centered feature matrix
            using var centeredX = new FeatureMatrix(rows, cols);
            float[] cRaw = centeredX.RawArray;
            float[] fRaw = features.RawArray;

            Parallel.For(0, rows, r =>
            {
                int rOffset = r * cols;
                for (int c = 0; c < cols; c++)
                {
                    cRaw[rOffset + c] = fRaw[rOffset + c] - meanX[c];
                }
            });

            // Form Gram Matrix G = X_c^T * X_c (size cols x cols) on GPU
            float[] gram = new float[cols * cols];
            GpuMlAccelerator.ComputeGramMatrix(centeredX, gram, _target);

            // Add Ridge regularization (alpha * I) + numerical conditioning to diagonal
            float reg = _alpha > 0f ? _alpha : 1e-6f;
            for (int i = 0; i < cols; i++)
            {
                gram[i * cols + i] += reg;
            }

            // Compute right-hand side vector v = X_c^T * y_c
            float[] v = new float[cols];
            for (int c = 0; c < cols; c++)
            {
                double dot = 0;
                for (int r = 0; r < rows; r++)
                {
                    float yCent = targets[r] - meanY;
                    dot += cRaw[r * cols + c] * yCent;
                }
                v[c] = (float)dot;
            }

            // Solve G * w = v
            SolveLinearSystem(gram, v, _weights, cols);

            // Intercept = meanY - w^T * meanX
            double wDotMeanX = 0;
            for (int c = 0; c < cols; c++)
            {
                wDotMeanX += _weights[c] * meanX[c];
            }
            _intercept = (float)(meanY - wDotMeanX);
        }
        else
        {
            // No intercept: G = X^T * X
            float[] gram = new float[cols * cols];
            GpuMlAccelerator.ComputeGramMatrix(features, gram, _target);

            float reg = _alpha > 0f ? _alpha : 1e-6f;
            for (int i = 0; i < cols; i++)
            {
                gram[i * cols + i] += reg;
            }

            float[] v = new float[cols];
            for (int c = 0; c < cols; c++)
            {
                double dot = 0;
                for (int r = 0; r < rows; r++)
                {
                    dot += features.GetValue(r, c) * targets[r];
                }
                v[c] = (float)dot;
            }

            SolveLinearSystem(gram, v, _weights, cols);
            _intercept = 0f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<float> predictions)
    {
        int rows = features.Rows;
        int cols = features.Columns;

        if (predictions.Length < rows)
            throw new ArgumentException("Predictions span too small.", nameof(predictions));

        // Use GPU feature projection if large enough
        if (rows >= 1024 && GpuMlAccelerator.IsGpuAvailable && _target != GpuTarget.Cpu)
        {
            float[] temp = GC.AllocateArray<float>(rows, pinned: true);
            GpuMlAccelerator.ProjectFeatures(features, _weights, temp, 1, _target);
            for (int r = 0; r < rows; r++)
            {
                predictions[r] = temp[r] + _intercept;
            }
            return;
        }

        // Parallel CPU SIMD fallback
        unsafe
        {
            fixed (float* pData = features.RawArray)
            fixed (float* pPred = predictions)
            fixed (float* pWeights = _weights)
            {
                float intercept = _intercept;
                for (int r = 0; r < rows; r++)
                {
                    float* pRow = pData + (r * cols);
                    pPred[r] = LinearKernels.DotProduct(new ReadOnlySpan<float>(pRow, cols), _weights) + intercept;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictRow(ReadOnlySpan<float> row)
    {
        return LinearKernels.DotProduct(row, _weights) + _intercept;
    }

    /// <summary>
    /// Solves symmetric positive-definite linear system A * x = b using Cholesky decomposition L * L^T * x = b.
    /// Falls back to Gaussian elimination if not strictly positive definite.
    /// </summary>
    private static void SolveLinearSystem(float[] a, float[] b, float[] x, int n)
    {
        // Try Cholesky
        float[] l = new float[n * n];
        bool choleskySuccess = true;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                float sum = 0f;
                for (int k = 0; k < j; k++)
                {
                    sum += l[i * n + k] * l[j * n + k];
                }

                if (i == j)
                {
                    float val = a[i * n + i] - sum;
                    if (val <= 1e-12f)
                    {
                        choleskySuccess = false;
                        break;
                    }
                    l[i * n + j] = MathF.Sqrt(val);
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
            // Forward solve L * y = b
            float[] y = new float[n];
            for (int i = 0; i < n; i++)
            {
                float sum = 0f;
                for (int k = 0; k < i; k++) sum += l[i * n + k] * y[k];
                y[i] = (b[i] - sum) / l[i * n + i];
            }

            // Back solve L^T * x = y
            for (int i = n - 1; i >= 0; i--)
            {
                float sum = 0f;
                for (int k = i + 1; k < n; k++) sum += l[k * n + i] * x[k];
                x[i] = (y[i] - sum) / l[i * n + i];
            }
            return;
        }

        // Gaussian elimination with partial pivoting fallback
        float[] mat = (float[])a.Clone();
        float[] rhs = (float[])b.Clone();

        for (int i = 0; i < n; i++)
        {
            int maxRow = i;
            float maxVal = MathF.Abs(mat[i * n + i]);
            for (int k = i + 1; k < n; k++)
            {
                float val = MathF.Abs(mat[k * n + i]);
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = k;
                }
            }

            if (maxRow != i)
            {
                for (int k = i; k < n; k++)
                {
                    (mat[i * n + k], mat[maxRow * n + k]) = (mat[maxRow * n + k], mat[i * n + k]);
                }
                (rhs[i], rhs[maxRow]) = (rhs[maxRow], rhs[i]);
            }

            float pivot = mat[i * n + i];
            if (MathF.Abs(pivot) < 1e-12f) pivot = 1e-6f;

            for (int k = i + 1; k < n; k++)
            {
                float factor = mat[k * n + i] / pivot;
                for (int j = i; j < n; j++)
                {
                    mat[k * n + j] -= factor * mat[i * n + j];
                }
                rhs[k] -= factor * rhs[i];
            }
        }

        for (int i = n - 1; i >= 0; i--)
        {
            float sum = 0f;
            for (int k = i + 1; k < n; k++) sum += mat[i * n + k] * x[k];
            float diag = mat[i * n + i];
            x[i] = (rhs[i] - sum) / (MathF.Abs(diag) > 1e-12f ? diag : 1e-6f);
        }
    }
}
