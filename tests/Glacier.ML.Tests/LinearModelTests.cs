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
}
