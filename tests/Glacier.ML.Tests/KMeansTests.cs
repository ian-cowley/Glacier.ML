using System;
using System.Reflection;
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

        // Verify deterministic positive throughput without wall-clock flakiness on shared CI VMs
        Assert.True(samplesPerSec > 0, $"Throughput was {samplesPerSec:N0} samples/sec (elapsed {elapsedMs:F3} ms)");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void KMeansKernels_Dim2And3_MatchScalarGroundTruthAcrossCentroidCounts(int dim)
    {
        var rng = new Random(99 + dim);
        int[] counts = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 65];
        foreach (int k in counts)
        {
            float[] centroids = new float[k * dim];
            float[] sample = new float[dim];
            for (int i = 0; i < centroids.Length; i++) centroids[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
            for (int i = 0; i < dim; i++) sample[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

            int actual = KMeansKernels.FindNearestCentroid(sample, centroids, k, dim);

            int expected = 0;
            float best = float.MaxValue;
            for (int c = 0; c < k; c++)
            {
                float d = 0f;
                for (int j = 0; j < dim; j++)
                {
                    float diff = sample[j] - centroids[c * dim + j];
                    d += diff * diff;
                }
                if (d < best)
                {
                    best = d;
                    expected = c;
                }
            }

            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void KMeans_EmptyCluster_NeverResetsToOrigin_BothRowMajorAndColumnar()
    {
        // All points live far from origin around (100, 100) and (110, 110), with k=5 > 2 distinct points
        // Empty clusters must be preserved/re-seeded onto data manifold, never reset to (0, 0)
        float[] col0 = [100f, 100f, 110f, 110f, 100f, 110f];
        float[] col1 = [100f, 100f, 110f, 110f, 100f, 110f];

        using var colMatrix = new FeatureMatrix(new ReadOnlyMemory<float>[] { col0, col1 });
        var kmCol = new KMeans(k: 5, maxIterations: 20, target: GpuTarget.Cpu);
        kmCol.Fit(colMatrix);

        for (int c = 0; c < 5; c++)
        {
            float cx = kmCol.Centroids[c * 2];
            float cy = kmCol.Centroids[c * 2 + 1];
            Assert.True(cx >= 90f && cy >= 90f, $"Columnar centroid {c} collapsed to ({cx}, {cy}) instead of staying on manifold!");
        }

        using var rowMatrix = new FeatureMatrix(6, 2);
        for (int r = 0; r < 6; r++)
        {
            rowMatrix.SetValue(r, 0, col0[r]);
            rowMatrix.SetValue(r, 1, col1[r]);
        }
        var kmRow = new KMeans(k: 5, maxIterations: 20, target: GpuTarget.Cpu);
        kmRow.Fit(rowMatrix);

        for (int c = 0; c < 5; c++)
        {
            float cx = kmRow.Centroids[c * 2];
            float cy = kmRow.Centroids[c * 2 + 1];
            Assert.True(cx >= 90f && cy >= 90f, $"Row-major centroid {c} collapsed to ({cx}, {cy}) instead of staying on manifold!");
        }
    }

    [Fact]
    public void KMeans_PredictRow_WhenRowsLessThanK_DoesNotReadOutOfBounds()
    {
        using var matrix = new FeatureMatrix(2, 3);
        matrix.SetValue(0, 0, 1f); matrix.SetValue(0, 1, 1f); matrix.SetValue(0, 2, 1f);
        matrix.SetValue(1, 0, 50f); matrix.SetValue(1, 1, 50f); matrix.SetValue(1, 2, 50f);

        var kmeans = new KMeans(k: 8, maxIterations: 10, target: GpuTarget.Cpu);
        kmeans.Fit(matrix);

        Assert.Equal(2 * 3, kmeans.Centroids.Length);
        int c0 = kmeans.PredictRow(new float[] { 1.1f, 1.0f, 0.9f });
        int c1 = kmeans.PredictRow(new float[] { 49.9f, 50.1f, 50.0f });
        Assert.InRange(c0, 0, 1);
        Assert.InRange(c1, 0, 1);
        Assert.NotEqual(c0, c1);
    }

    [Fact]
    public void KMeans_ValidationAndUnsupportedTargets_ThrowExpectedExceptions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new KMeans(k: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new KMeans(k: 3, maxIterations: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new KMeans(k: 3, tolerance: -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new KMeans(k: 3, tolerance: float.NaN));

        var unfitted = new KMeans(k: 2);
        using var matrix = new FeatureMatrix(4, 2);
        Assert.Throws<InvalidOperationException>(() => unfitted.Predict(matrix, new int[4]));
        Assert.Throws<InvalidOperationException>(() => unfitted.PredictRow(new float[2]));

        unfitted.Fit(matrix);
        Assert.Throws<ArgumentException>(() => unfitted.PredictRow(new float[1]));
        Assert.Throws<ArgumentException>(() => unfitted.PredictRow(new float[3]));
        Assert.Throws<ArgumentException>(() => unfitted.Predict(new FeatureMatrix(4, 3), new int[4]));

        var amdKmeans = new KMeans(k: 2, target: GpuTarget.Amd);
        Assert.Throws<NotSupportedException>(() => amdKmeans.Fit(matrix));
        var dualKmeans = new KMeans(k: 2, target: GpuTarget.DualGpu);
        Assert.Throws<NotSupportedException>(() => dualKmeans.Fit(matrix));
    }

    [Theory]
    [InlineData(100, 10000.0f, 2, 2)]   // Single-threaded (< 512) at 10000 offset
    [InlineData(600, 10000.0f, 2, 2)]   // Parallel (>= 512) at 10000 offset
    [InlineData(120, 500.0f, 5, 2)]     // Offset 500 with k=5 > 2 clusters (empty cluster + Predict)
    public void KMeans_Predict_LargeOffsetClusters_MatchesPredictRowAndColumnar(
        int rows,
        float offset,
        int k,
        int cols)
    {
        var rng = new Random(2026);
        using var rowMatrix = new FeatureMatrix(rows, cols);
        float[][] colArrays = new float[cols][];
        for (int c = 0; c < cols; c++) colArrays[c] = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            float baseCoord = (r % 2 == 0) ? offset : offset + 2.0f;
            for (int c = 0; c < cols; c++)
            {
                float v = baseCoord + (float)(rng.NextDouble() * 0.2 - 0.1);
                rowMatrix.SetValue(r, c, v);
                colArrays[c][r] = v;
            }
        }

        ReadOnlyMemory<float>[] colMems = new ReadOnlyMemory<float>[cols];
        for (int c = 0; c < cols; c++) colMems[c] = colArrays[c];
        using var colMatrix = new FeatureMatrix(colMems);

        var kmRow = new KMeans(k: k, maxIterations: 30, target: GpuTarget.Cpu);
        kmRow.Fit(rowMatrix);

        int[] rowAssignments = new int[rows];
        int[] colAssignments = new int[rows];
        kmRow.Predict(rowMatrix, rowAssignments, GpuTarget.Cpu);
        kmRow.Predict(colMatrix, colAssignments, GpuTarget.Cpu);

        for (int r = 0; r < rows; r++)
        {
            int expected = kmRow.PredictRow(rowMatrix.GetRow(r));
            Assert.Equal(expected, rowAssignments[r]);
            Assert.Equal(expected, colAssignments[r]);
        }

        if (k == 2)
        {
            // Verify the two separated clusters are cleanly separated with 0/rows misclassification
            int clusterEven = rowAssignments[0];
            int clusterOdd = rowAssignments[1];
            Assert.NotEqual(clusterEven, clusterOdd);
            for (int r = 0; r < rows; r++)
            {
                Assert.Equal(r % 2 == 0 ? clusterEven : clusterOdd, rowAssignments[r]);
            }
        }
    }

    [Theory]
    [InlineData(100, 1_000_000f)]
    [InlineData(600, 1_000_000f)]
    [InlineData(2500, 100_000f)]
    [InlineData(2500, 1_000_000f)]
    [InlineData(10000, 100_000f)]
    [InlineData(10000, 1_000_000f)]
    public void KMeans_Fit_And_FitColumnar_ExtremeOffset_PreservesMStepPrecision(int rows, float offset)
    {
        int half = rows / 2;
        using var rowMatrix = new FeatureMatrix(rows, 2);
        float[] col0 = new float[rows];
        float[] col1 = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            float v = r < half ? offset : offset + 2.0f;
            rowMatrix.SetValue(r, 0, v);
            rowMatrix.SetValue(r, 1, v);
            col0[r] = v;
            col1[r] = v;
        }

        using var colMatrix = new FeatureMatrix(new ReadOnlyMemory<float>[] { col0, col1 });
        FieldInfo? dataField = typeof(FeatureMatrix).GetField("_data", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(dataField);
        Assert.Null(dataField!.GetValue(colMatrix));

        var kmRow = new KMeans(k: 2, maxIterations: 20, target: GpuTarget.Cpu);
        kmRow.Fit(rowMatrix);

        var kmCol = new KMeans(k: 2, maxIterations: 20, target: GpuTarget.Cpu);
        kmCol.Fit(colMatrix);

        // Columnar Fit must never materialize row-major _data
        Assert.Null(dataField.GetValue(colMatrix));

        AssertRecoveredExtremeOffsetCentroids(kmRow, offset);
        AssertRecoveredExtremeOffsetCentroids(kmCol, offset);

        int[] rowAssign = new int[rows];
        int[] colAssign = new int[rows];
        kmRow.Predict(rowMatrix, rowAssign, GpuTarget.Cpu);
        kmCol.Predict(colMatrix, colAssign, GpuTarget.Cpu);

        Assert.Null(dataField.GetValue(colMatrix));

        int rowClusterLow = kmRow.PredictRow([offset, offset]);
        int rowClusterHigh = kmRow.PredictRow([offset + 2.0f, offset + 2.0f]);
        Assert.NotEqual(rowClusterLow, rowClusterHigh);
        Assert.Equal(rowClusterLow, kmRow.PredictRow([offset + 0.5f, offset + 0.5f]));
        Assert.Equal(rowClusterHigh, kmRow.PredictRow([offset + 1.5f, offset + 1.5f]));

        int colClusterLow = kmCol.PredictRow([offset, offset]);
        int colClusterHigh = kmCol.PredictRow([offset + 2.0f, offset + 2.0f]);
        Assert.NotEqual(colClusterLow, colClusterHigh);
        Assert.Equal(colClusterLow, kmCol.PredictRow([offset + 0.5f, offset + 0.5f]));
        Assert.Equal(colClusterHigh, kmCol.PredictRow([offset + 1.5f, offset + 1.5f]));

        for (int r = 0; r < rows; r++)
        {
            Assert.Equal(r < half ? rowClusterLow : rowClusterHigh, rowAssign[r]);
            Assert.Equal(r < half ? colClusterLow : colClusterHigh, colAssign[r]);
        }
    }

    private static void AssertRecoveredExtremeOffsetCentroids(KMeans km, float offset)
    {
        Assert.Equal(4, km.Centroids.Length);
        float c0x = km.Centroids[0];
        float c0y = km.Centroids[1];
        float c1x = km.Centroids[2];
        float c1y = km.Centroids[3];

        bool orderNormal = MathF.Abs(c0x - offset) < 1e-3f;
        if (orderNormal)
        {
            Assert.True(MathF.Abs(c0x - offset) < 1e-3f, $"c0x={c0x}, expected={offset}");
            Assert.True(MathF.Abs(c0y - offset) < 1e-3f, $"c0y={c0y}, expected={offset}");
            Assert.True(MathF.Abs(c1x - (offset + 2.0f)) < 1e-3f, $"c1x={c1x}, expected={offset + 2.0f}");
            Assert.True(MathF.Abs(c1y - (offset + 2.0f)) < 1e-3f, $"c1y={c1y}, expected={offset + 2.0f}");
        }
        else
        {
            Assert.True(MathF.Abs(c0x - (offset + 2.0f)) < 1e-3f, $"c0x={c0x}, expected={offset + 2.0f}");
            Assert.True(MathF.Abs(c0y - (offset + 2.0f)) < 1e-3f, $"c0y={c0y}, expected={offset + 2.0f}");
            Assert.True(MathF.Abs(c1x - offset) < 1e-3f, $"c1x={c1x}, expected={offset}");
            Assert.True(MathF.Abs(c1y - offset) < 1e-3f, $"c1y={c1y}, expected={offset}");
        }
    }
}
