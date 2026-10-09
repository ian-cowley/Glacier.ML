using System;
using Glacier.ML.Clustering;
using Glacier.ML.Core;
using Xunit;

namespace Glacier.ML.Tests;

public class KMeansTests
{
    [Fact]
    public void EuclideanDistanceSquared_CalculatesCorrectly()
    {
        ReadOnlySpan<float> p1 = stackalloc float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f };
        ReadOnlySpan<float> p2 = stackalloc float[] { 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f };

        float dist = KMeansKernels.EuclideanDistanceSquared(p1, p2);
        // Each coordinate difference is 1, 1^2 = 1, sum across 9 coordinates = 9
        Assert.Equal(9f, dist, precision: 4);
    }

    [Fact]
    public void KMeans_FindsSeparatedClusters()
    {
        const int pointsPerCluster = 100;
        const int totalPoints = pointsPerCluster * 3;
        var matrix = new FeatureMatrix(totalPoints, 2);

        var rng = new Random(42);

        // Cluster 0: Centered at (0, 0)
        for (int i = 0; i < pointsPerCluster; i++)
        {
            matrix.SetValue(i, 0, (float)(rng.NextDouble() * 2 - 1));
            matrix.SetValue(i, 1, (float)(rng.NextDouble() * 2 - 1));
        }

        // Cluster 1: Centered at (50, 50)
        for (int i = 0; i < pointsPerCluster; i++)
        {
            matrix.SetValue(pointsPerCluster + i, 0, 50f + (float)(rng.NextDouble() * 2 - 1));
            matrix.SetValue(pointsPerCluster + i, 1, 50f + (float)(rng.NextDouble() * 2 - 1));
        }

        // Cluster 2: Centered at (-50, 50)
        for (int i = 0; i < pointsPerCluster; i++)
        {
            matrix.SetValue(2 * pointsPerCluster + i, 0, -50f + (float)(rng.NextDouble() * 2 - 1));
            matrix.SetValue(2 * pointsPerCluster + i, 1, 50f + (float)(rng.NextDouble() * 2 - 1));
        }

        var kmeans = new KMeans(k: 3, maxIterations: 50);
        kmeans.Fit(matrix);

        int[] assignments = new int[totalPoints];
        kmeans.Predict(matrix, assignments);

        // Check intra-cluster consistency
        int c0 = assignments[0];
        int c1 = assignments[pointsPerCluster];
        int c2 = assignments[2 * pointsPerCluster];

        // All 3 centroids must be distinct
        Assert.NotEqual(c0, c1);
        Assert.NotEqual(c1, c2);
        Assert.NotEqual(c0, c2);

        // All points in cluster 0 should belong to c0
        for (int i = 0; i < pointsPerCluster; i++)
        {
            Assert.Equal(c0, assignments[i]);
        }

        // All points in cluster 1 should belong to c1
        for (int i = 0; i < pointsPerCluster; i++)
        {
            Assert.Equal(c1, assignments[pointsPerCluster + i]);
        }

        // All points in cluster 2 should belong to c2
        for (int i = 0; i < pointsPerCluster; i++)
        {
            Assert.Equal(c2, assignments[2 * pointsPerCluster + i]);
        }

        // Test PredictRow
        ReadOnlySpan<float> query0 = stackalloc float[] { 0.2f, -0.1f };
        ReadOnlySpan<float> query1 = stackalloc float[] { 50.1f, 49.9f };
        ReadOnlySpan<float> query2 = stackalloc float[] { -49.8f, 50.2f };

        Assert.Equal(c0, kmeans.PredictRow(query0));
        Assert.Equal(c1, kmeans.PredictRow(query1));
        Assert.Equal(c2, kmeans.PredictRow(query2));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(16)]
    public void KMeansKernels_FindNearestCentroid_MatchesScalar(int dim)
    {
        int k = 4;
        float[] centroids = new float[k * dim];
        var rng = new Random(123);
        for (int i = 0; i < centroids.Length; i++) centroids[i] = (float)rng.NextDouble() * 10f;

        float[] sample = new float[dim];
        for (int i = 0; i < dim; i++) sample[i] = (float)rng.NextDouble() * 10f;

        int nearest = KMeansKernels.FindNearestCentroid(sample, centroids, k, dim);

        // Compute expected via scalar reference
        int expectedCluster = 0;
        float minD = float.MaxValue;
        for (int c = 0; c < k; c++)
        {
            float d = 0f;
            for (int j = 0; j < dim; j++)
            {
                float diff = sample[j] - centroids[c * dim + j];
                d += diff * diff;
            }
            if (d < minD)
            {
                minD = d;
                expectedCluster = c;
            }
        }

        Assert.Equal(expectedCluster, nearest);
    }

    [Fact]
    public void KMeans_MultiThreadedMStep_RunsCorrectlyOnLargeDataset()
    {
        const int rows = 5000;
        const int cols = 8;
        const int k = 4;

        var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(42);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                matrix.SetValue(r, c, (float)rng.NextDouble() * 100f);
            }
        }

        var kmeans = new KMeans(k: k, maxIterations: 10);
        kmeans.Fit(matrix);

        int[] assignments = new int[rows];
        kmeans.Predict(matrix, assignments);

        for (int r = 0; r < rows; r++)
        {
            Assert.InRange(assignments[r], 0, k - 1);
        }
    }

    [Theory]
    [InlineData(2, 50)]
    [InlineData(3, 50)]
    [InlineData(4, 100)]
    [InlineData(8, 200)]
    [InlineData(10, 300)]
    [InlineData(16, 400)]
    [InlineData(20, 600)]
    [InlineData(32, 1000)]
    public void PredictBatchDecomposed_MatchesFindNearestCentroid_AcrossDimensions(int dim, int numSamples)
    {
        const int k = 5;
        var rng = new Random(42 + dim);

        float[] centroids = new float[k * dim];
        for (int i = 0; i < centroids.Length; i++) centroids[i] = (float)rng.NextDouble() * 100f;

        float[] centroidNormsSq = new float[k];
        for (int c = 0; c < k; c++)
        {
            float norm = 0f;
            for (int d = 0; d < dim; d++)
            {
                float val = centroids[c * dim + d];
                norm += val * val;
            }
            centroidNormsSq[c] = norm;
        }

        float[] data = new float[numSamples * dim];
        for (int i = 0; i < data.Length; i++) data[i] = (float)rng.NextDouble() * 100f;

        int[] expectedAssignments = new int[numSamples];
        int[] actualAssignments = new int[numSamples];

        // Scalar reference via FindNearestCentroid
        for (int r = 0; r < numSamples; r++)
        {
            var rowSpan = data.AsSpan(r * dim, dim);
            expectedAssignments[r] = KMeansKernels.FindNearestCentroid(rowSpan, centroids, k, dim);
        }

        // Vectorized batch decomposed
        KMeansKernels.PredictBatchDecomposed(
            data,
            centroids,
            centroidNormsSq,
            0,
            numSamples,
            dim,
            k,
            actualAssignments);

        for (int r = 0; r < numSamples; r++)
        {
            Assert.Equal(expectedAssignments[r], actualAssignments[r]);
        }
    }

    [Fact]
    public void KMeans_CentroidNormsSq_CalculatedCorrectly()
    {
        const int rows = 200;
        const int cols = 6;
        const int k = 3;

        var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(789);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                matrix.SetValue(r, c, (float)rng.NextDouble() * 10f);
            }
        }

        var kmeans = new KMeans(k: k, maxIterations: 15);
        kmeans.Fit(matrix);

        Assert.Equal(k, kmeans.CentroidNormsSq.Length);
        for (int c = 0; c < k; c++)
        {
            float expectedNorm = 0f;
            int offset = c * cols;
            for (int d = 0; d < cols; d++)
            {
                float val = kmeans.Centroids[offset + d];
                expectedNorm += val * val;
            }
            Assert.Equal(expectedNorm, kmeans.CentroidNormsSq[c], precision: 3);
        }
    }

    [Fact]
    public void KMeans_Predict_SmallAndLargeBatches_ProducesConsistentResults()
    {
        const int rows = 1200;
        const int cols = 8;
        const int k = 4;

        var trainMatrix = new FeatureMatrix(rows, cols);
        var rng = new Random(999);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                trainMatrix.SetValue(r, c, (float)rng.NextDouble() * 50f);
            }
        }

        var kmeans = new KMeans(k: k, maxIterations: 10);
        kmeans.Fit(trainMatrix);

        // Test small batch (100 rows, < 512)
        var smallMatrix = new FeatureMatrix(100, cols);
        for (int r = 0; r < 100; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                smallMatrix.SetValue(r, c, trainMatrix.GetValue(r, c));
            }
        }
        int[] smallAssignments = new int[100];
        kmeans.Predict(smallMatrix, smallAssignments);

        // Test full batch (1200 rows, >= 512, parallel path)
        int[] largeAssignments = new int[rows];
        kmeans.Predict(trainMatrix, largeAssignments);

        // First 100 rows must match exactly between small and large batch predictions
        for (int r = 0; r < 100; r++)
        {
            Assert.Equal(smallAssignments[r], largeAssignments[r]);
        }
    }

    [Fact]
    public void KMeans_Predict_50k_Samples_HighThroughput()
    {
        const int nSamples = 50_000;
        const int nFeatures = 10;
        const int k = 5;

        using var matrix = new FeatureMatrix(nSamples, nFeatures);
        var rng = new Random(42);
        for (int i = 0; i < nSamples; i++)
        {
            for (int c = 0; c < nFeatures; c++)
            {
                matrix.SetValue(i, c, (float)rng.NextDouble() * 10f);
            }
        }

        var kmeans = new KMeans(k: k, maxIterations: 10, target: GpuTarget.Cpu);
        kmeans.Fit(matrix);

        int[] assignments = new int[nSamples];

        // Warmup
        for (int i = 0; i < 3; i++)
        {
            kmeans.Predict(matrix, assignments);
        }

        // Measure 10 runs
        var sw = System.Diagnostics.Stopwatch.StartNew();
        const int runs = 10;
        for (int r = 0; r < runs; r++)
        {
            kmeans.Predict(matrix, assignments);
        }
        sw.Stop();

        double elapsedMs = sw.Elapsed.TotalMilliseconds / runs;
        double samplesPerSec = nSamples / (elapsedMs / 1000.0);

        // Verify valid assignments
        for (int i = 0; i < nSamples; i++)
        {
            Assert.InRange(assignments[i], 0, k - 1);
        }

        // Throughput must exceed 50M samples/sec on standard hardware (target > 80M samples/sec).
        // On virtualized, low-core CI environments (e.g. 2-vCPU GitHub Actions runners), adjust threshold.
        double minExpected = (Environment.ProcessorCount <= 4 || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CI")) || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("GITHUB_ACTIONS")))
            ? 25_000_000
            : 50_000_000;
        Assert.True(samplesPerSec > minExpected, $"Throughput was {samplesPerSec:N0} samples/sec, expected > {minExpected:N0} samples/sec (elapsed {elapsedMs:F3} ms)");
    }
}
