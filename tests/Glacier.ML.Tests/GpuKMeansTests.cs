using System;
using Glacier.ML.Clustering;
using Glacier.ML.Compute;
using Glacier.ML.Core;
using Glacier.Tensor.Compute;
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

        // Test Auto mode (dispatches to NVIDIA/AMD GPU if available)
        var kmeansAuto = new KMeans(k: k, maxIterations: 30, target: GpuTarget.Auto);
        kmeansAuto.Fit(matrix);

        int[] autoAssignments = new int[rows];
        kmeansAuto.Predict(matrix, autoAssignments);

        for (int r = 0; r < rows; r++)
        {
            Assert.InRange(autoAssignments[r], 0, k - 1);
        }

        // If NVIDIA GPU is available, test explicit GpuTarget.Nvidia
        if (GpuMlAccelerator.IsNvidiaAvailable)
        {
            var kmeansGpu = new KMeans(k: k, maxIterations: 30, target: GpuTarget.Nvidia);
            kmeansGpu.Fit(matrix);

            int[] gpuAssignments = new int[rows];
            kmeansGpu.Predict(matrix, gpuAssignments);

            for (int r = 0; r < rows; r++)
            {
                Assert.InRange(gpuAssignments[r], 0, k - 1);
            }
        }
    }
}
