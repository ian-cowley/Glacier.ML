using System;
using Glacier.ML.Core;
using Glacier.ML.Linear;
using Xunit;

namespace Glacier.ML.Tests;

public class LinearModelTests
{
    [Fact]
    public void DotProduct_CalculatesAccurately()
    {
        ReadOnlySpan<float> a = stackalloc float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
        ReadOnlySpan<float> b = stackalloc float[] { 2f, 0.5f, 1f, 2f, 1f, 0.5f, 2f, 1f };
        // 2 + 1 + 3 + 8 + 5 + 3 + 14 + 8 = 44
        float dot = LinearKernels.DotProduct(a, b);
        Assert.Equal(44f, dot, precision: 4);
    }

    [Fact]
    public void Sigmoid_EvaluatesCorrectValues()
    {
        Assert.Equal(0.5f, LinearKernels.Sigmoid(0f), precision: 4);
        Assert.True(LinearKernels.Sigmoid(10f) > 0.999f);
        Assert.True(LinearKernels.Sigmoid(-10f) < 0.001f);
    }

    [Fact]
    public void LogisticRegression_ConvergesOnLinearlySeparableData()
    {
        const int n = 300;
        var matrix = new FeatureMatrix(n, 2);
        float[] targets = new float[n];

        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            float x1 = (float)rng.NextDouble() * 10f - 5f; // [-5, 5]
            float x2 = (float)rng.NextDouble() * 10f - 5f; // [-5, 5]
            matrix.SetValue(i, 0, x1);
            matrix.SetValue(i, 1, x2);
            // Decision boundary: 2*x1 - 3*x2 + 1 > 0
            targets[i] = (2f * x1 - 3f * x2 + 1f > 0f) ? 1.0f : 0.0f;
        }

        var model = new FastLogisticRegression(learningRate: 0.1f, epochs: 100);
        model.Fit(matrix, targets);

        float[] predictions = new float[n];
        model.Predict(matrix, predictions);

        float accuracy = Metrics.Accuracy(targets, predictions);
        Assert.True(accuracy >= 0.90f, $"Logistic regression accuracy was {accuracy}, expected >= 0.90");

        // Verify PredictProbability
        ReadOnlySpan<float> strongPositive = stackalloc float[] { 4f, -4f };
        ReadOnlySpan<float> strongNegative = stackalloc float[] { -4f, 4f };

        Assert.True(model.PredictProbability(strongPositive) > 0.7f);
        Assert.True(model.PredictProbability(strongNegative) < 0.3f);
    }

    [Fact]
    public void FastLogisticRegression_ValidationAndColumnarZeroRawArray_Works()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FastLogisticRegression(learningRate: 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FastLogisticRegression(learningRate: float.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FastLogisticRegression(epochs: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FastLogisticRegression(l2Regularization: -0.1f));

        var unfitted = new FastLogisticRegression();
        using var rowMat = new FeatureMatrix(4, 2);
        Assert.Throws<InvalidOperationException>(() => unfitted.Predict(rowMat, new float[4]));
        Assert.Throws<InvalidOperationException>(() => unfitted.PredictRow(new float[2]));
        Assert.Throws<InvalidOperationException>(() => unfitted.PredictProbability(new float[2]));

        // Columnar SGD (< 1024) + Columnar Batch (>= 1024) must keep _data == null
        var dataField = typeof(FeatureMatrix).GetField("_data", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        const int smallN = 100;
        float[] sc0 = new float[smallN];
        float[] sc1 = new float[smallN];
        float[] sy = new float[smallN];
        for (int i = 0; i < smallN; i++)
        {
            sc0[i] = i < 50 ? -2f : 2f;
            sc1[i] = i < 50 ? -1f : 1f;
            sy[i] = i < 50 ? 0f : 1f;
        }
        using var smallCol = new FeatureMatrix(new ReadOnlyMemory<float>[] { sc0, sc1 });
        var logSmall = new FastLogisticRegression(learningRate: 0.2f, epochs: 30);
        logSmall.Fit(smallCol, sy);
        Assert.Null(dataField?.GetValue(smallCol));

        float[] smallPreds = new float[smallN];
        logSmall.Predict(smallCol, smallPreds);
        Assert.Null(dataField?.GetValue(smallCol));
        Assert.Equal(1.0f, Metrics.Accuracy(sy, smallPreds), precision: 3);

        // Short / mismatched vectors must throw ArgumentException instead of silently truncating
        Assert.Throws<ArgumentException>(() => logSmall.PredictRow(new float[] { 1.0f }));
        Assert.Throws<ArgumentException>(() => logSmall.PredictProbability(new float[] { 1.0f }));
        Assert.Throws<ArgumentException>(() => logSmall.PredictRow(new float[] { 1.0f, 2.0f, 3.0f }));
        Assert.Throws<ArgumentException>(() => logSmall.Predict(new FeatureMatrix(4, 3), new float[4]));

        const int largeN = 1200;
        float[] lc0 = new float[largeN];
        float[] lc1 = new float[largeN];
        float[] ly = new float[largeN];
        for (int i = 0; i < largeN; i++)
        {
            lc0[i] = i % 2 == 0 ? -3f : 3f;
            lc1[i] = i % 2 == 0 ? -1f : 1f;
            ly[i] = i % 2 == 0 ? 0f : 1f;
        }
        using var largeCol = new FeatureMatrix(new ReadOnlyMemory<float>[] { lc0, lc1 });
        var logLarge = new FastLogisticRegression(learningRate: 0.2f, epochs: 30);
        logLarge.Fit(largeCol, ly);
        Assert.Null(dataField?.GetValue(largeCol));

        float[] largePreds = new float[largeN];
        logLarge.Predict(largeCol, largePreds);
        Assert.Null(dataField?.GetValue(largeCol));
        Assert.Equal(1.0f, Metrics.Accuracy(ly, largePreds), precision: 3);

        Assert.Throws<NotSupportedException>(() => new FastLogisticRegression(target: GpuTarget.Nvidia).Fit(rowMat, new float[4]));
    }
}
