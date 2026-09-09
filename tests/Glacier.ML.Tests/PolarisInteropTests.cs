using System;
using Glacier.ML.Interop;
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
}
