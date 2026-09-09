using System;
using Glacier.ML.Core;
using Glacier.ML.Preprocessing;
using Xunit;

namespace Glacier.ML.Tests;

public class PreprocessorTests
{
    [Fact]
    public void StandardScaler_StandardizesToZeroMeanAndUnitVariance()
    {
        const int n = 1000;
        var matrix = new FeatureMatrix(n, 2);

        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            // Col 0: mean 10, std ~ 2
            matrix.SetValue(i, 0, 10f + (float)(rng.NextDouble() * 4 - 2));
            // Col 1: mean 50, std ~ 10
            matrix.SetValue(i, 1, 50f + (float)(rng.NextDouble() * 20 - 10));
        }

        var scaler = new StandardScaler();
        scaler.Fit(matrix);
        scaler.Transform(matrix);

        // Check means are ~0 and stds are ~1
        double sum0 = 0, sum1 = 0;
        for (int i = 0; i < n; i++)
        {
            sum0 += matrix.GetValue(i, 0);
            sum1 += matrix.GetValue(i, 1);
        }

        Assert.True(Math.Abs(sum0 / n) < 0.05);
        Assert.True(Math.Abs(sum1 / n) < 0.05);
    }

    [Fact]
    public void MinMaxScaler_ScalesToZeroOne()
    {
        const int n = 50;
        var matrix = new FeatureMatrix(n, 2);

        for (int i = 0; i < n; i++)
        {
            matrix.SetValue(i, 0, i);            // 0 to 49
            matrix.SetValue(i, 1, 100f + i * 2); // 100 to 198
        }

        var scaler = new MinMaxScaler();
        scaler.Fit(matrix);
        scaler.Transform(matrix);

        Assert.Equal(0.0f, matrix.GetValue(0, 0), precision: 4);
        Assert.Equal(1.0f, matrix.GetValue(49, 0), precision: 4);
        Assert.Equal(0.0f, matrix.GetValue(0, 1), precision: 4);
        Assert.Equal(1.0f, matrix.GetValue(49, 1), precision: 4);
    }

    [Fact]
    public void OneHotEncoder_EncodesProperBinaryVectors()
    {
        ReadOnlySpan<int> categories = stackalloc int[] { 10, 20, 10, 30, 20 };
        var matrix = new FeatureMatrix(5, 3);

        var encoder = new OneHotEncoder();
        encoder.Fit(categories);

        Assert.Equal(3, encoder.NumCategories);

        encoder.Transform(categories, matrix, startCol: 0);

        // Category 10 -> index 0
        Assert.Equal(1.0f, matrix.GetValue(0, 0));
        Assert.Equal(0.0f, matrix.GetValue(0, 1));
        Assert.Equal(0.0f, matrix.GetValue(0, 2));

        // Category 20 -> index 1
        Assert.Equal(0.0f, matrix.GetValue(1, 0));
        Assert.Equal(1.0f, matrix.GetValue(1, 1));
        Assert.Equal(0.0f, matrix.GetValue(1, 2));

        // Category 30 -> index 2
        Assert.Equal(0.0f, matrix.GetValue(3, 0));
        Assert.Equal(0.0f, matrix.GetValue(3, 1));
        Assert.Equal(1.0f, matrix.GetValue(3, 2));
    }
}
