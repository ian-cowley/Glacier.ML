using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Compute;
using Glacier.ML.Core;
using Glacier.Tensor.Compute;

namespace Glacier.ML.Decomposition;

/// <summary>
/// Hardware-accelerated Principal Component Analysis (PCA) for .NET 10.
/// Solves covariance matrix and feature projection on GPU (NVIDIA RTX 4060 dGPU, AMD Radeon 890M APU, Auto, and CPU).
/// Equivalent to Scikit-Learn sklearn.decomposition.PCA.
/// </summary>
public sealed class FastPCA
{
    private readonly int _nComponents;
    private readonly GpuTarget _target;
    private float[] _components = Array.Empty<float>(); // [nComponents, nFeatures]
    private float[] _mean = Array.Empty<float>();       // [nFeatures]
    private float[] _explainedVariance = Array.Empty<float>();
    private float[] _explainedVarianceRatio = Array.Empty<float>();
    private int _nFeatures;

    public int NComponents => _nComponents;
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

    /// <summary>
    /// Fits the PCA model on the feature matrix X.
    /// Computes the column means, builds the covariance matrix on GPU via Gram matrix GEMM,
    /// and solves for the principal components.
    /// </summary>
    public void Fit(FeatureMatrix features)
    {
        int rows = features.Rows;
        int cols = features.Columns;
        _nFeatures = cols;
        int k = Math.Min(_nComponents, cols);

        // 1. Calculate column means
        _mean = new float[cols];
        for (int c = 0; c < cols; c++)
        {
            double sum = 0;
            for (int r = 0; r < rows; r++)
            {
                sum += features.GetValue(r, c);
            }
            _mean[c] = (float)(sum / rows);
        }

        // 2. Create centered feature matrix
        using var centered = new FeatureMatrix(rows, cols);
        float[] cRaw = centered.RawArray;
        float[] fRaw = features.RawArray;

        Parallel.For(0, rows, r =>
        {
            int rOffset = r * cols;
            for (int c = 0; c < cols; c++)
            {
                cRaw[rOffset + c] = fRaw[rOffset + c] - _mean[c];
            }
        });

        // 3. Compute Covariance Matrix: Cov = (1 / (rows - 1)) * X_centered^T * X_centered
        float[] cov = new float[cols * cols];
        GpuMlAccelerator.ComputeGramMatrix(centered, cov, _target);

        float factor = 1.0f / Math.Max(1, rows - 1);
        double totalVariance = 0;
        for (int i = 0; i < cols; i++)
        {
            totalVariance += cov[i * cols + i] * factor;
        }

        for (int i = 0; i < cov.Length; i++)
        {
            cov[i] *= factor;
        }

        // 4. Extract top k eigenvectors & eigenvalues using Power Iteration with Gram-Schmidt orthogonalization
        _components = new float[k * cols];
        _explainedVariance = new float[k];
        _explainedVarianceRatio = new float[k];

        var rng = new Random(42);

        for (int c = 0; c < k; c++)
        {
            float[] v = new float[cols];
            for (int i = 0; i < cols; i++) v[i] = (float)rng.NextDouble() - 0.5f;

            // Normalize initial random vector
            Normalize(v);

            float eigenvalue = 0f;
            float[] w = new float[cols];

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
            {
                // w = Cov * v
                for (int i = 0; i < cols; i++)
                {
                    float sum = 0f;
                    int iOffset = i * cols;
                    for (int j = 0; j < cols; j++)
                    {
                        sum += cov[iOffset + j] * v[j];
                    }
                    w[i] = sum;
                }

                // Deflation: Orthogonalize w against all previously found eigenvectors
                for (int prev = 0; prev < c; prev++)
                {
                    ReadOnlySpan<float> prevVec = _components.AsSpan(prev * cols, cols);
                    float dot = Dot(w, prevVec);
                    for (int j = 0; j < cols; j++)
                    {
                        w[j] -= dot * prevVec[j];
                    }
                }

                // Compute norm
                eigenvalue = Norm(w);
                if (eigenvalue < 1e-8f) break;

                // Normalize
                float invNorm = 1.0f / eigenvalue;
                float diff = 0f;
                for (int j = 0; j < cols; j++)
                {
                    float nextV = w[j] * invNorm;
                    float delta = nextV - v[j];
                    diff += delta * delta;
                    v[j] = nextV;
                }

                if (diff < 1e-10f) break; // Converged
            }

            v.CopyTo(_components.AsSpan(c * cols, cols));
            _explainedVariance[c] = eigenvalue;
            _explainedVarianceRatio[c] = totalVariance > 0 ? (float)(eigenvalue / totalVariance) : 0f;
        }
    }

    /// <summary>
    /// Projects feature matrix X onto the principal components: Y = (X - mean) * W^T.
    /// GPU accelerated via GpuAccelerator matrix multiplication.
    /// </summary>
    public void Transform(FeatureMatrix features, Span<float> destination)
    {
        int rows = features.Rows;
        int cols = features.Columns;
        int k = _nComponents;

        if (cols != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {cols}.", nameof(features));
        if (destination.Length < rows * k)
            throw new ArgumentException($"Destination span too small. Required {rows * k}, got {destination.Length}.", nameof(destination));

        // Center features
        using var centered = new FeatureMatrix(rows, cols);
        float[] cRaw = centered.RawArray;
        float[] fRaw = features.RawArray;

        Parallel.For(0, rows, r =>
        {
            int rOffset = r * cols;
            for (int c = 0; c < cols; c++)
            {
                cRaw[rOffset + c] = fRaw[rOffset + c] - _mean[c];
            }
        });

        // Components is [k, cols]. For Y = X_c * W^T, we transpose W so W^T is [cols, k]
        float[] wT = GC.AllocateArray<float>(cols * k, pinned: true);
        for (int c = 0; c < k; c++)
        {
            for (int d = 0; d < cols; d++)
            {
                wT[d * k + c] = _components[c * cols + d];
            }
        }

        GpuMlAccelerator.ProjectFeatures(centered, wT, destination, k, _target);
    }

    /// <summary>
    /// Reconstructs original data from the reduced dimension space: X_reconstructed = Y * W + mean.
    /// </summary>
    public void InverseTransform(ReadOnlySpan<float> reduced, Span<float> destination, int rows)
    {
        int cols = _nFeatures;
        int k = _nComponents;

        if (reduced.Length < rows * k)
            throw new ArgumentException("Reduced span too small.", nameof(reduced));
        if (destination.Length < rows * cols)
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
                    int rOffset = r * cols;
                    int redOffset = r * k;

                    for (int d = 0; d < cols; d++)
                    {
                        float sum = pMean[d];
                        for (int c = 0; c < k; c++)
                        {
                            sum += pRed[redOffset + c] * pComp[c * cols + d];
                        }
                        pDest[rOffset + d] = sum;
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Norm(ReadOnlySpan<float> v)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++) sum += v[i] * v[i];
        return MathF.Sqrt(sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Normalize(Span<float> v)
    {
        float norm = Norm(v);
        if (norm > 1e-12f)
        {
            float inv = 1.0f / norm;
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }
    }
}
