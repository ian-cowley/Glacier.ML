using System;
using Glacier.ML.Clustering;
using Glacier.ML.Compute;
using Glacier.ML.Core;
using Xunit;

namespace Glacier.ML.Tests;

public class GpuKMeansTests
{
    [Fact]
    public void KMeans_FitAndPredict_CpuAndGpuTargets_Converges()
    {
        const int rows = 1000;
        const int cols = 8;
        const int k = 4;

        using var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(42);
        for (int r = 0; r < rows; r++)
        {
            int cluster = r % k;
            for (int c = 0; c < cols; c++)
            {
                // Each cluster is centered around cluster * 20
                float val = cluster * 20.0f + (float)(rng.NextDouble() * 5.0);
                matrix.SetValue(r, c, val);
            }
        }

        // Test CPU mode
        var kmeansCpu = new KMeans(k: k, maxIterations: 30, target: GpuTarget.Cpu);
        kmeansCpu.Fit(matrix);

        Assert.Equal(k, kmeansCpu.K);
        Assert.Equal(k * cols, kmeansCpu.Centroids.Length);

        int[] cpuAssignments = new int[rows];
        kmeansCpu.Predict(matrix, cpuAssignments);

        // Verify all rows have a valid cluster assignment [0, k-1]
        for (int r = 0; r < rows; r++)
        {
            Assert.InRange(cpuAssignments[r], 0, k - 1);
        }

        // Test Auto mode (uses NVIDIA GPU if available, otherwise SIMD CPU)
        var kmeansAuto = new KMeans(k: k, maxIterations: 30, target: GpuTarget.Auto);
        kmeansAuto.Fit(matrix);

        int[] autoAssignments = new int[rows];
        kmeansAuto.Predict(matrix, autoAssignments);

        for (int r = 0; r < rows; r++)
        {
            Assert.InRange(autoAssignments[r], 0, k - 1);
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void KMeans_FitAndPredict_NvidiaTarget_ExecutesOnGpu()
    {
        if (!GpuMlAccelerator.IsNvidiaAvailable) return;

        const int rows = 1000;
        const int cols = 8;
        const int k = 4;

        using var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(42);
        for (int r = 0; r < rows; r++)
        {
            int cluster = r % k;
            for (int c = 0; c < cols; c++)
            {
                matrix.SetValue(r, c, cluster * 20.0f + (float)(rng.NextDouble() * 5.0));
            }
        }

        var kmeansGpu = new KMeans(k: k, maxIterations: 30, target: GpuTarget.Nvidia);
        kmeansGpu.Fit(matrix);

        int[] gpuAssignments = new int[rows];
        kmeansGpu.Predict(matrix, gpuAssignments);

        for (int r = 0; r < rows; r++)
        {
            Assert.InRange(gpuAssignments[r], 0, k - 1);
        }
    }

    [Theory]
    [InlineData(GpuTarget.Amd)]
    [InlineData(GpuTarget.DualGpu)]
    public void GpuMlAccelerator_UnsupportedGpuTargets_ThrowNotSupportedException(GpuTarget target)
    {
        float[] samples = new float[16];
        float[] centroids = new float[8];
        int[] assignments = new int[4];
        float[] a = new float[64];
        float[] b = new float[64];
        float[] dest = new float[64];

        Assert.Throws<NotSupportedException>(() =>
            GpuMlAccelerator.AssignClustersGpu(samples, centroids, assignments, 4, 2, 4, target));
        Assert.Throws<NotSupportedException>(() =>
            GpuMlAccelerator.VectorAdd(a, b, dest, target));
        Assert.Throws<NotSupportedException>(() =>
            GpuMlAccelerator.VectorFma(a, b, dest, 2.0f, target));

        using var matrix = new FeatureMatrix(4, 2);
        float[] gram = new float[4];
        float[] weights = new float[4];
        float[] proj = new float[8];
        Assert.Throws<NotSupportedException>(() =>
            GpuMlAccelerator.ComputeGramMatrix(matrix, gram, target));
        Assert.Throws<NotSupportedException>(() =>
            GpuMlAccelerator.ProjectFeatures(matrix, weights, proj, 2, target));
    }

    [Fact]
    public void GpuMlAccelerator_CpuOnlyMatrixOps_ThrowNotSupportedForExplicitNvidia()
    {
        using var matrix = new FeatureMatrix(4, 2);
        float[] gram = new float[4];
        float[] weights = new float[4];
        float[] proj = new float[8];

        Assert.Throws<NotSupportedException>(() =>
            GpuMlAccelerator.ComputeGramMatrix(matrix, gram, GpuTarget.Nvidia));
        Assert.Throws<NotSupportedException>(() =>
            GpuMlAccelerator.ProjectFeatures(matrix, weights, proj, 2, GpuTarget.Nvidia));
    }
}
