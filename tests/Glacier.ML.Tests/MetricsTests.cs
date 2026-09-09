using System;
using Glacier.ML.Core;
using Xunit;

namespace Glacier.ML.Tests;

public class MetricsTests
{
    [Fact]
    public void Accuracy_ComputesCorrectRatio()
    {
        ReadOnlySpan<float> actual = stackalloc float[] { 1f, 0f, 1f, 1f, 0f };
        ReadOnlySpan<float> pred = stackalloc float[] { 1f, 0f, 0f, 1f, 0f };

        float acc = Metrics.Accuracy(actual, pred);
        Assert.Equal(0.8f, acc, precision: 4);
    }

    [Fact]
    public void MeanSquaredError_ComputesZeroForPerfectPredictions()
    {
        ReadOnlySpan<float> actual = stackalloc float[] { 1.5f, 2.5f, 3.5f, 4.5f };
        ReadOnlySpan<float> pred = stackalloc float[] { 1.5f, 2.5f, 3.5f, 4.5f };

        float mse = Metrics.MeanSquaredError(actual, pred);
        Assert.Equal(0.0f, mse, precision: 4);
    }

    [Fact]
    public void MeanSquaredError_ComputesCorrectValue()
    {
        ReadOnlySpan<float> actual = stackalloc float[] { 1f, 2f, 3f, 4f };
        ReadOnlySpan<float> pred = stackalloc float[] { 2f, 3f, 4f, 5f };

        // Each diff is 1, squared is 1, sum is 4, mean is 1
        float mse = Metrics.MeanSquaredError(actual, pred);
        Assert.Equal(1.0f, mse, precision: 4);
    }

    [Fact]
    public void R2Score_ComputesAccurateScores()
    {
        ReadOnlySpan<float> actual = stackalloc float[] { 3f, -0.5f, 2f, 7f };
        ReadOnlySpan<float> pred = stackalloc float[] { 2.5f, 0.0f, 2f, 8f };

        float r2 = Metrics.R2Score(actual, pred);
        Assert.True(r2 > 0.9f);
    }
}
