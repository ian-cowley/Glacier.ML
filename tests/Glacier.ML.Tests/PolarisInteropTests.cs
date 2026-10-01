using System;
using Glacier.ML.Core;
using Glacier.ML.Interop;
using Glacier.ML.Trees;
using Glacier.Polaris;
using Glacier.Polaris.Data;
using Xunit;

namespace Glacier.ML.Tests;

public class PolarisInteropTests
{
    [Fact]
    public void DataFrame_ToFeatureMatrix_ExtractsNumericColumnsAccurately()
    {
        var colX1 = new Float64Series("f1", new double[] { 1.5, 2.5, 3.5 });
        var colX2 = new Int32Series("f2", new int[] { 10, 20, 30 });
        var df = new DataFrame(new ISeries[] { colX1, colX2 });

        var matrix = df.ToFeatureMatrix("f1", "f2");

        Assert.Equal(3, matrix.Rows);
        Assert.Equal(2, matrix.Columns);

        Assert.Equal(1.5f, matrix.GetValue(0, 0));
        Assert.Equal(10.0f, matrix.GetValue(0, 1));
        Assert.Equal(2.5f, matrix.GetValue(1, 0));
        Assert.Equal(20.0f, matrix.GetValue(1, 1));
        Assert.Equal(3.5f, matrix.GetValue(2, 0));
        Assert.Equal(30.0f, matrix.GetValue(2, 1));
    }

    [Fact]
    public void DataFrame_ToFeatureMatrix_ZeroCopy_SharesMemoryDirectly()
    {
        var col1 = new Float32Series("c1", 3);
        col1.Memory.Span[0] = 10f;
        col1.Memory.Span[1] = 20f;
        col1.Memory.Span[2] = 30f;

        var col2 = new Float32Series("c2", 3);
        col2.Memory.Span[0] = 100f;
        col2.Memory.Span[1] = 200f;
        col2.Memory.Span[2] = 300f;

        var df = new DataFrame(new ISeries[] { col1, col2 });
        var matrix = df.ToFeatureMatrix("c1", "c2");

        Assert.True(matrix.IsColumnar);
        Assert.Equal(10f, matrix.GetColumn(0)[0]);
        Assert.Equal(100f, matrix.GetColumn(1)[0]);

        // Mutate original series memory buffer directly:
        col1.Memory.Span[0] = 999.0f;
        col2.Memory.Span[0] = 888.0f;

        // Zero-copy verification: change must be immediately reflected in FeatureMatrix
        Assert.Equal(999.0f, matrix.GetColumn(0)[0]);
        Assert.Equal(999.0f, matrix.GetValue(0, 0));
        Assert.Equal(888.0f, matrix.GetColumn(1)[0]);
        Assert.Equal(888.0f, matrix.GetValue(0, 1));
    }

    [Fact]
    public void DataFrame_FitRandomForest_And_FitKMeans_Succeeds()
    {
        var colX1 = new Float64Series("x1", new double[] { 0.1, 0.2, 0.3, 5.1, 5.2, 5.3 });
        var colX2 = new Float64Series("x2", new double[] { 0.2, 0.1, 0.3, 5.0, 5.1, 5.2 });
        var colLabel = new Int32Series("label", new int[] { 0, 0, 0, 1, 1, 1 });

        var df = new DataFrame(new ISeries[] { colX1, colX2, colLabel });

        var forest = df.FitRandomForest("label", new[] { "x1", "x2" }, numTrees: 5, maxDepth: 3);
        Assert.NotNull(forest);

        var matrix = df.ToFeatureMatrix("x1", "x2");
        float[] preds = new float[6];
        forest.Predict(matrix, preds);

        Assert.Equal(0.0f, preds[0]);
        Assert.Equal(1.0f, preds[3]);

        var kmeans = df.FitKMeans(new[] { "x1", "x2" }, k: 2, maxIterations: 10);
        Assert.NotNull(kmeans);

        int[] assignments = new int[6];
        kmeans.Predict(matrix, assignments);

        Assert.Equal(assignments[0], assignments[1]);
        Assert.Equal(assignments[0], assignments[2]);
        Assert.Equal(assignments[3], assignments[4]);
        Assert.Equal(assignments[3], assignments[5]);
        Assert.NotEqual(assignments[0], assignments[3]);
    }

    [Fact]
    public void DataFrame_FitRandomForest_ZeroCopy_AssertZeroHeapArrayCopies()
    {
        const int rows = 500;
        var rng = new Random(42);

        var col0 = new Float32Series("f0", rows);
        var col1 = new Float32Series("f1", rows);
        var targetCol = new Float32Series("label", rows);

        for (int r = 0; r < rows; r++)
        {
            float v0 = (float)rng.NextDouble() * 10f;
            float v1 = (float)rng.NextDouble() * 10f;
            col0.Memory.Span[r] = v0;
            col1.Memory.Span[r] = v1;
            targetCol.Memory.Span[r] = (v0 + v1 > 10.0f) ? 1.0f : 0.0f;
        }

        var df = new DataFrame(new ISeries[] { col0, col1, targetCol });
        var matrix = df.ToFeatureMatrix("f0", "f1");

        // Verify zero-copy columnar backing: _data managed array must be null
        Assert.True(matrix.IsColumnar);
        var dataField = typeof(FeatureMatrix).GetField("_data", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.Null(dataField?.GetValue(matrix));

        var forest = df.FitRandomForest("label", new[] { "f0", "f1" }, numTrees: 10, maxDepth: 5);
        Assert.NotNull(forest);

        // Assert that even after fitting, _data remains null (zero heap matrix array allocations)
        Assert.Null(dataField?.GetValue(matrix));

        float[] predictions = new float[rows];
        forest.Predict(matrix, predictions);

        float accuracy = Metrics.Accuracy(targetCol.Memory.Span, predictions);
        Assert.True(accuracy >= 0.85f, $"Accuracy was {accuracy}, expected >= 0.85");
    }

    [Fact]
    public void DataFrame_FitKMeans_ZeroCopy_DirectColumnarExecution()
    {
        const int rows = 300;
        var colA = new Float32Series("a", rows);
        var colB = new Float32Series("b", rows);

        // 3 clusters around (1, 1), (10, 10), (20, 20)
        for (int r = 0; r < 100; r++)
        {
            colA.Memory.Span[r] = 1.0f + r * 0.01f;
            colB.Memory.Span[r] = 1.0f - r * 0.01f;

            colA.Memory.Span[100 + r] = 10.0f + r * 0.01f;
            colB.Memory.Span[100 + r] = 10.0f - r * 0.01f;

            colA.Memory.Span[200 + r] = 20.0f + r * 0.01f;
            colB.Memory.Span[200 + r] = 20.0f - r * 0.01f;
        }

        var df = new DataFrame(new ISeries[] { colA, colB });
        var matrix = df.ToFeatureMatrix("a", "b");
        Assert.True(matrix.IsColumnar);

        var dataField = typeof(FeatureMatrix).GetField("_data", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.Null(dataField?.GetValue(matrix));

        var kmeans = df.FitKMeans(new[] { "a", "b" }, k: 3, maxIterations: 20);
        Assert.NotNull(kmeans);

        // Assert that after KMeans training, _data remains null (zero heap matrix array allocations)
        Assert.Null(dataField?.GetValue(matrix));

        int[] assignments = new int[rows];
        kmeans.Predict(matrix, assignments);

        // Cluster 0 points should share an assignment, cluster 1 points should share an assignment
        Assert.Equal(assignments[0], assignments[50]);
        Assert.Equal(assignments[100], assignments[150]);
        Assert.Equal(assignments[200], assignments[250]);
        Assert.NotEqual(assignments[0], assignments[100]);
        Assert.NotEqual(assignments[100], assignments[200]);
    }

    [Fact]
    public void DataFrame_FitPCA_ZeroCopy_ExtractsPrincipalComponents()
    {
        const int rows = 200;
        var colX = new Float32Series("x", rows);
        var colY = new Float32Series("y", rows);
        var colZ = new Float32Series("z", rows);

        var rng = new Random(123);
        for (int r = 0; r < rows; r++)
        {
            float t = (float)rng.NextDouble() * 5f;
            colX.Memory.Span[r] = t * 2.0f;
            colY.Memory.Span[r] = t * -1.0f;
            colZ.Memory.Span[r] = (float)rng.NextDouble() * 0.1f;
        }

        var df = new DataFrame(new ISeries[] { colX, colY, colZ });
        var matrix = df.ToFeatureMatrix("x", "y", "z");
        Assert.True(matrix.IsColumnar);

        var pca = df.FitPCA(nComponents: 2, "x", "y", "z");
        Assert.NotNull(pca);
        Assert.Equal(2, pca.NComponents);
        Assert.Equal(3, pca.NFeatures);

        // Check descending variance
        Assert.True(pca.ExplainedVariance[0] >= pca.ExplainedVariance[1]);

        // Transform features directly over columnar matrix
        float[] reduced = new float[rows * 2];
        pca.Transform(matrix, reduced);

        for (int i = 0; i < rows * 2; i++)
        {
            Assert.False(float.IsNaN(reduced[i]));
        }
    }
}
