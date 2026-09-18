namespace Glacier.ML.Tests;

using System;
using System.Threading.Tasks;
using Glacier.ML.Compute;
using Glacier.ML.Core;
using Glacier.ML.Trees;
using Xunit;

public class AdversarialMlTests
{
    [Fact]
    public void GpuMlStreamContext_ConcurrentMultiThreadedStressTest()
    {
        // Concurrently rent, resize, and return stream contexts across 32 threads
        const int numTasks = 32;
        const int iterationsPerTask = 25;

        Parallel.For(0, numTasks, taskId =>
        {
            for (int iter = 0; iter < iterationsPerTask; iter++)
            {
                nuint samplesBytes = (nuint)((taskId * 1024 + iter * 128) * sizeof(float));
                nuint centroidsBytes = (nuint)(16 * 8 * sizeof(float));
                nuint assignBytes = (nuint)(1024 * sizeof(int));

                using var ctx = new GpuMlStreamContext();
                ctx.EnsureCapacity(samplesBytes, centroidsBytes, assignBytes);

                Assert.True(ctx.CapSamples >= samplesBytes);
                Assert.True(ctx.CapCentroids >= centroidsBytes);
                Assert.True(ctx.CapAssignments >= assignBytes);
            }
        });
    }

    [Fact]
    public void FastDecisionTree_InPlaceHoarePartitioning_AccurateSplitsAndZeroRecursiveAllocations()
    {
        const int n = 600;
        const int cols = 4;
        using var matrix = new FeatureMatrix(n, cols);
        float[] targets = new float[n];

        var rng = new Random(777);
        for (int i = 0; i < n; i++)
        {
            float f0 = (float)rng.NextDouble() * 10f;
            float f1 = (float)rng.NextDouble() * 10f;
            float f2 = (float)rng.NextDouble() * 10f;
            float f3 = (float)rng.NextDouble() * 10f;

            matrix.SetValue(i, 0, f0);
            matrix.SetValue(i, 1, f1);
            matrix.SetValue(i, 2, f2);
            matrix.SetValue(i, 3, f3);

            // Non-linear checkerboard/interval decision boundary:
            // class 1 if (f0 > 5 && f1 < 5) OR (f0 <= 5 && f2 > 5)
            bool c1 = (f0 > 5.0f && f1 < 5.0f) || (f0 <= 5.0f && f2 > 5.0f);
            targets[i] = c1 ? 1.0f : 0.0f;
        }

        var tree = new FastDecisionTree(maxDepth: 8, minSamplesSplit: 2);
        tree.Fit(matrix, targets);

        float[] predictions = new float[n];
        tree.Predict(matrix, predictions);

        float accuracy = Metrics.Accuracy(targets, predictions);
        Assert.True(accuracy >= 0.95f, $"Expected accuracy >= 0.95, got {accuracy}");

        // Boundary stress test 1: All identical targets
        float[] allZeros = new float[50];
        using var matZeros = new FeatureMatrix(50, 2);
        var treeZeros = new FastDecisionTree(maxDepth: 4);
        treeZeros.Fit(matZeros, allZeros);
        Span<float> predZeros = stackalloc float[50];
        treeZeros.Predict(matZeros, predZeros);
        for (int i = 0; i < 50; i++) Assert.Equal(0.0f, predZeros[i]);

        // Boundary stress test 2: All identical features with mixed targets
        using var matIdentical = new FeatureMatrix(50, 2);
        float[] mixedTargets = new float[50];
        for (int i = 0; i < 50; i++) mixedTargets[i] = i < 30 ? 1.0f : 0.0f; // majority 1.0
        var treeIdentical = new FastDecisionTree(maxDepth: 4);
        treeIdentical.Fit(matIdentical, mixedTargets);
        Span<float> predIdentical = stackalloc float[50];
        treeIdentical.Predict(matIdentical, predIdentical);
        for (int i = 0; i < 50; i++) Assert.Equal(1.0f, predIdentical[i]);
    }
}
