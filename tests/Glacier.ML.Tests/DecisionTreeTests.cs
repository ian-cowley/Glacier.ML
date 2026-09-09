using System;
using Glacier.ML.Core;
using Glacier.ML.Trees;
using Xunit;

namespace Glacier.ML.Tests;

public class DecisionTreeTests
{
    [Fact]
    public void DecisionTree_SeparatesLinearlySeparableData()
    {
        const int n = 200;
        var matrix = new FeatureMatrix(n, 2);
        float[] targets = new float[n];

        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            float x1 = (float)rng.NextDouble() * 10f;
            float x2 = (float)rng.NextDouble() * 10f;
            matrix.SetValue(i, 0, x1);
            matrix.SetValue(i, 1, x2);
            // Class 1 if x1 > 5.0, else 0
            targets[i] = (x1 > 5.0f) ? 1.0f : 0.0f;
        }

        var tree = new FastDecisionTree(maxDepth: 4);
        tree.Fit(matrix, targets);

        float[] predictions = new float[n];
        tree.Predict(matrix, predictions);

        float accuracy = Metrics.Accuracy(targets, predictions);
        Assert.True(accuracy >= 0.98f, $"Expected accuracy >= 0.98, got {accuracy}");

        // Test single row prediction
        ReadOnlySpan<float> testRowLow = stackalloc float[] { 2.0f, 8.0f };
        ReadOnlySpan<float> testRowHigh = stackalloc float[] { 8.0f, 2.0f };

        Assert.Equal(0.0f, tree.PredictRow(testRowLow));
        Assert.Equal(1.0f, tree.PredictRow(testRowHigh));
    }

    [Fact]
    public void DecisionTree_HandlesZeroAllocationPredict()
    {
        var matrix = new FeatureMatrix(20, 2);
        float[] targets = new float[20];
        for (int i = 0; i < 20; i++)
        {
            matrix.SetValue(i, 0, i);
            matrix.SetValue(i, 1, 10f);
            targets[i] = i >= 10 ? 1.0f : 0.0f;
        }

        var tree = new FastDecisionTree(maxDepth: 3);
        tree.Fit(matrix, targets);

        Span<float> preds = stackalloc float[20];
        tree.Predict(matrix, preds);

        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(targets[i], preds[i]);
        }
    }
}
