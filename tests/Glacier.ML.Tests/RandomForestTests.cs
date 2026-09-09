using System;
using Glacier.ML.Core;
using Glacier.ML.Trees;
using Xunit;

namespace Glacier.ML.Tests;

public class RandomForestTests
{
    [Fact]
    public void RandomForest_AchievesHighAccuracy_OnSyntheticData()
    {
        const int n = 500;
        var matrix = new FeatureMatrix(n, 4);
        float[] targets = new float[n];

        var rng = new Random(1337);
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

            // True function: f0 + f1 > 10.0
            targets[i] = (f0 + f1 > 10.0f) ? 1.0f : 0.0f;
        }

        var forest = new FastRandomForest(numTrees: 30, maxDepth: 6, subsampleRatio: 0.8f);
        forest.Fit(matrix, targets);

        float[] predictions = new float[n];
        forest.Predict(matrix, predictions);

        float accuracy = Metrics.Accuracy(targets, predictions);
        Assert.True(accuracy >= 0.90f, $"Random Forest accuracy was {accuracy}, expected >= 0.90");

        // Check probability bounds
        float prob = forest.PredictProbability(matrix.GetRow(0));
        Assert.InRange(prob, 0.0f, 1.0f);
    }
}
