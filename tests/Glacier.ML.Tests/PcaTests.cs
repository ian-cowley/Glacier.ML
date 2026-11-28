using System;
using Glacier.ML.Core;
using Glacier.ML.Decomposition;
using Xunit;

namespace Glacier.ML.Tests;

public class PcaTests
{
    [Fact]
    public void FastPCA_FitAndTransform_ComponentsOrthogonalAndOrdered()
    {
        const int rows = 500;
        const int cols = 6;
        const int k = 3;

        using var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(42);
        for (int r = 0; r < rows; r++)
        {
            float t = (float)rng.NextDouble() * 10f;
            matrix.SetValue(r, 0, t * 2.0f + (float)rng.NextDouble());
            matrix.SetValue(r, 1, t * -1.5f + (float)rng.NextDouble());
            matrix.SetValue(r, 2, (float)rng.NextDouble() * 0.5f);
            matrix.SetValue(r, 3, (float)rng.NextDouble() * 0.2f);
            matrix.SetValue(r, 4, t * 0.8f + (float)rng.NextDouble());
            matrix.SetValue(r, 5, (float)rng.NextDouble() * 0.1f);
        }

        var pca = new FastPCA(nComponents: k, target: GpuTarget.Auto);
        pca.Fit(matrix);

        Assert.Equal(k, pca.NComponents);
        Assert.Equal(cols, pca.NFeatures);
        Assert.Equal(k, pca.ExplainedVariance.Length);

        // Explained variance should be strictly non-negative and descending
        for (int i = 0; i < k; i++)
        {
            Assert.True(pca.ExplainedVariance[i] >= 0f, $"Variance at {i} should be >= 0");
            if (i > 0)
            {
                Assert.True(pca.ExplainedVariance[i - 1] >= pca.ExplainedVariance[i] - 1e-4f,
                    $"Variance at {i-1} ({pca.ExplainedVariance[i-1]}) should be >= at {i} ({pca.ExplainedVariance[i]})");
            }
        }

        // Test Orthogonality of components: dot(c_i, c_j) ~ 0 for i != j, ~ 1 for i == j
        for (int i = 0; i < k; i++)
        {
            ReadOnlySpan<float> compI = pca.Components.AsSpan(i * cols, cols);
            float normSq = 0f;
            for (int d = 0; d < cols; d++) normSq += compI[d] * compI[d];
            Assert.True(MathF.Abs(normSq - 1.0f) < 0.05f, $"Component {i} should be normalized, got normSq={normSq}");

            for (int j = i + 1; j < k; j++)
            {
                ReadOnlySpan<float> compJ = pca.Components.AsSpan(j * cols, cols);
                float dot = 0f;
                for (int d = 0; d < cols; d++) dot += compI[d] * compJ[d];
                Assert.True(MathF.Abs(dot) < 0.05f, $"Components {i} and {j} should be orthogonal, got dot={dot}");
            }
        }

        // Transform features
        float[] reduced = new float[rows * k];
        pca.Transform(matrix, reduced);

        // Inverse transform and check reconstruction
        float[] reconstructed = new float[rows * cols];
        pca.InverseTransform(reduced, reconstructed, rows);

        // Verify reconstructed output is finite
        for (int i = 0; i < rows * cols; i++)
        {
            Assert.False(float.IsNaN(reconstructed[i]));
            Assert.False(float.IsInfinity(reconstructed[i]));
        }
    }

    [Fact]
    public void FastPCA_DegenerateRepeatedEigenvalues_JacobiAchievesTightOrthogonality()
    {
        // Construct data with degenerate / closely-spaced variances across 4 axes
        const int rows = 400;
        const int cols = 4;
        using var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(17);

        for (int r = 0; r < rows; r++)
        {
            // Orthogonal Rademacher-like isotropic components so eigenvalues are nearly equal (~1.0)
            float u0 = (r % 2 == 0 ? 1f : -1f) + (float)(rng.NextDouble() * 0.01);
            float u1 = ((r / 2) % 2 == 0 ? 1f : -1f) + (float)(rng.NextDouble() * 0.01);
            float u2 = ((r / 4) % 2 == 0 ? 1f : -1f) + (float)(rng.NextDouble() * 0.01);
            float u3 = ((r / 8) % 2 == 0 ? 0.5f : -0.5f) + (float)(rng.NextDouble() * 0.01);
            matrix.SetValue(r, 0, u0);
            matrix.SetValue(r, 1, u1);
            matrix.SetValue(r, 2, u2);
            matrix.SetValue(r, 3, u3);
        }

        var pca = new FastPCA(nComponents: cols, target: GpuTarget.Cpu);
        pca.Fit(matrix);

        for (int i = 0; i < cols; i++)
        {
            ReadOnlySpan<float> ci = pca.Components.AsSpan(i * cols, cols);
            float normSq = 0f;
            for (int d = 0; d < cols; d++) normSq += ci[d] * ci[d];
            Assert.True(MathF.Abs(normSq - 1.0f) < 1e-5f, $"NormSq[{i}]={normSq}");

            for (int j = i + 1; j < cols; j++)
            {
                ReadOnlySpan<float> cj = pca.Components.AsSpan(j * cols, cols);
                float dot = 0f;
                for (int d = 0; d < cols; d++) dot += ci[d] * cj[d];
                Assert.True(MathF.Abs(dot) < 1e-5f, $"Dot({i},{j})={dot}");
            }

            if (i > 0)
            {
                Assert.True(pca.ExplainedVariance[i - 1] >= pca.ExplainedVariance[i]);
            }
        }
    }

    [Fact]
    public void FastPCA_NComponentsExceedsColumns_ClampsSafelyAndInverseTransforms()
    {
        const int rows = 20;
        const int cols = 4;
        using var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(101);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                matrix.SetValue(r, c, (float)rng.NextDouble() * 10f);

        var pca = new FastPCA(nComponents: 10, target: GpuTarget.Auto);
        pca.Fit(matrix);

        Assert.Equal(cols, pca.NComponents);
        Assert.Equal(cols, pca.ExplainedVariance.Length);

        float[] reduced = new float[rows * pca.NComponents];
        pca.Transform(matrix, reduced);

        float[] reconstructed = new float[rows * cols];
        pca.InverseTransform(reduced, reconstructed, rows);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                Assert.Equal(matrix.GetValue(r, c), reconstructed[r * cols + c], precision: 3);
            }
        }
    }

    [Fact]
    public void FastPCA_ValidationAndUnsupportedTargets_ThrowExpectedExceptions()
    {
        var unfitted = new FastPCA(nComponents: 2);
        using var matrix = new FeatureMatrix(5, 3);
        Assert.Throws<InvalidOperationException>(() => unfitted.Transform(matrix, new float[10]));
        Assert.Throws<InvalidOperationException>(() => unfitted.InverseTransform(new float[10], new float[15], 5));

        unfitted.Fit(matrix);
        Assert.Throws<ArgumentOutOfRangeException>(() => unfitted.InverseTransform(new float[10], new float[15], -1));

        Assert.Throws<NotSupportedException>(() => new FastPCA(2, GpuTarget.Nvidia).Fit(matrix));
        Assert.Throws<NotSupportedException>(() => new FastPCA(2, GpuTarget.Amd).Fit(matrix));
        Assert.Throws<NotSupportedException>(() => new FastPCA(2, GpuTarget.DualGpu).Fit(matrix));
    }
}
