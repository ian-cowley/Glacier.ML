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

    [Fact]
    public void DecisionTree_InPlacePartitioning_MultiFeatureAndSubsets_Correct()
    {
        const int n = 500;
        const int cols = 5;
        var matrix = new FeatureMatrix(n, cols);
        float[] targets = new float[n];

        var rng = new Random(999);
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < cols; c++)
            {
                matrix.SetValue(i, c, (float)rng.NextDouble() * 100f);
            }
            // Class depends on feature 2 and feature 4
            float f2 = matrix.GetValue(i, 2);
            float f4 = matrix.GetValue(i, 4);
            targets[i] = (f2 > 50f && f4 < 70f) ? 1.0f : 0.0f;
        }

        var treeAll = new FastDecisionTree(maxDepth: 6, minSamplesSplit: 4);
        treeAll.Fit(matrix, targets);

        float[] predsAll = new float[n];
        treeAll.Predict(matrix, predsAll);
        float accAll = Metrics.Accuracy(targets, predsAll);
        Assert.True(accAll >= 0.90f, $"Expected high accuracy, got {accAll}");

        // Test with explicit feature subset (only features 2 and 4 active)
        int[] activeSubset = [2, 4];
        var treeSubset = new FastDecisionTree(maxDepth: 6, minSamplesSplit: 4);
        treeSubset.Fit(matrix, targets, activeSubset);

        float[] predsSubset = new float[n];
        treeSubset.Predict(matrix, predsSubset);
        float accSubset = Metrics.Accuracy(targets, predsSubset);
        Assert.True(accSubset >= 0.90f, $"Expected high accuracy on subset, got {accSubset}");
    }

    [Fact]
    public void SplitFinder_SkewedDataAndDuplicateBootstrapIndices_FindsOptimalSplit()
    {
        // Skewed: 990 zeros (class 0), 10 large outliers (class 1), plus duplicate bootstrap indices
        const int n = 1000;
        float[] featureValues = new float[n];
        float[] targets = new float[n];
        for (int i = 0; i < n; i++)
        {
            if (i >= 990)
            {
                featureValues[i] = 1000f + i;
                targets[i] = 1.0f;
            }
            else
            {
                featureValues[i] = 0f;
                targets[i] = 0.0f;
            }
        }

        int[] sampleIndices = new int[n];
        var rng = new Random(7);
        for (int i = 0; i < n; i++)
        {
            // Ensure both classes appear with duplicates
            sampleIndices[i] = i < 980 ? rng.Next(0, 990) : 990 + (i % 10);
        }

        var (threshold, gain) = SplitFinder.FindBestSplitClassification(featureValues, targets, sampleIndices);
        Assert.True(gain > 0.01f, $"Expected positive Gini gain on skewed data, got {gain}");
        Assert.True(threshold >= 0f && threshold < 1990f, $"Threshold {threshold} should separate 0 from outliers");
    }

    [Fact]
    public void SplitFinder_PureOrIdenticalFeatures_ReturnsZeroGain()
    {
        float[] vals = [5f, 5f, 5f, 5f, 5f, 5f];
        float[] targets = [0f, 1f, 0f, 1f, 0f, 1f];
        int[] indices = [0, 1, 2, 3, 4, 5];

        var (_, gainIdentical) = SplitFinder.FindBestSplitClassification(vals, targets, indices);
        Assert.Equal(0f, gainIdentical);

        float[] distinctVals = [1f, 2f, 3f, 4f, 5f, 6f];
        float[] pureTargets = [1f, 1f, 1f, 1f, 1f, 1f];
        var (_, gainPure) = SplitFinder.FindBestSplitClassification(distinctVals, pureTargets, indices);
        Assert.Equal(0f, gainPure);
    }

    [Fact]
    public void DecisionTreeAndRandomForest_BoundsAndUnfittedValidation_ThrowsExpectedExceptions()
    {
        var tree = new FastDecisionTree(maxDepth: 3);
        var forest = new FastRandomForest(numTrees: 5, maxDepth: 3);
        using var matrix = new FeatureMatrix(10, 2);
        float[] shortTargets = new float[5];
        float[] shortPreds = new float[5];
        float[] validTargets = new float[10];

        Assert.Throws<InvalidOperationException>(() => tree.Predict(matrix, new float[10]));
        Assert.Throws<InvalidOperationException>(() => tree.PredictRow(new float[2]));
        Assert.Throws<InvalidOperationException>(() => forest.Predict(matrix, new float[10]));
        Assert.Throws<InvalidOperationException>(() => forest.PredictRow(new float[2]));
        Assert.Throws<InvalidOperationException>(() => forest.PredictProbability(new float[2]));

        Assert.Throws<ArgumentException>(() => tree.Fit(matrix, shortTargets));
        Assert.Throws<ArgumentException>(() => forest.Fit(matrix, shortTargets));

        tree.Fit(matrix, validTargets);
        forest.Fit(matrix, validTargets);

        Assert.Throws<ArgumentException>(() => tree.Predict(matrix, shortPreds));
        Assert.Throws<ArgumentException>(() => tree.PredictRow(new float[1]));
        Assert.Throws<ArgumentException>(() => forest.Predict(matrix, shortPreds));
        Assert.Throws<ArgumentException>(() => forest.PredictRow(new float[3]));
        Assert.Throws<ArgumentException>(() => forest.PredictProbability(new float[1]));
    }

    [Fact]
    public void FastDecisionTree_And_SplitFinder_RouteNaNConsistently_SeparableNaNVsFinite()
    {
        // Challenger BUG-2 reproduction: 40 NaN (target 1.0), 30 1.0f (target 0.0), 30 2.0f (target 0.0)
        const int n = 100;
        float[] col0 = new float[n];
        float[] targets = new float[n];
        int[] indices = new int[n];
        using var rowMatrix = new FeatureMatrix(n, 1);

        for (int i = 0; i < n; i++)
        {
            float val = i < 40 ? float.NaN : (i < 70 ? 1.0f : 2.0f);
            float tgt = i < 40 ? 1.0f : 0.0f;
            col0[i] = val;
            targets[i] = tgt;
            indices[i] = i;
            rowMatrix.SetValue(i, 0, val);
        }

        var (threshold, gain) = SplitFinder.FindBestSplitClassification(col0, targets, indices);
        Assert.Equal(2.0f, threshold);
        Assert.Equal(0.48f, gain, precision: 4);

        var treeRow = new FastDecisionTree(maxDepth: 2, minSamplesSplit: 2);
        treeRow.Fit(rowMatrix, targets);
        float[] predsRow = new float[n];
        treeRow.Predict(rowMatrix, predsRow);
        Assert.Equal(1.0f, Metrics.Accuracy(targets, predsRow));

        using var colMatrix = new FeatureMatrix(new ReadOnlyMemory<float>[] { col0 });
        var treeCol = new FastDecisionTree(maxDepth: 2, minSamplesSplit: 2);
        treeCol.Fit(colMatrix, targets);
        float[] predsCol = new float[n];
        treeCol.Predict(colMatrix, predsCol);
        Assert.Equal(1.0f, Metrics.Accuracy(targets, predsCol));
    }

    [Theory]
    [InlineData(20, 12)]   // count <= 32 (FindBestSplitExactSmall)
    [InlineData(100, 40)]  // count > 32 (FindBestSplitHistogram)
    public void FastDecisionTree_And_SplitFinder_SingleFiniteValuePlusNaN_SeparatesCleanly(int totalCount, int nanCount)
    {
        float[] col0 = new float[totalCount];
        float[] targets = new float[totalCount];
        int[] indices = new int[totalCount];
        using var matrix = new FeatureMatrix(totalCount, 1);

        for (int i = 0; i < totalCount; i++)
        {
            bool isNan = i < nanCount;
            float val = isNan ? float.NaN : 5.0f;
            float tgt = isNan ? 1.0f : 0.0f;
            col0[i] = val;
            targets[i] = tgt;
            indices[i] = i;
            matrix.SetValue(i, 0, val);
        }

        var (threshold, gain) = SplitFinder.FindBestSplitClassification(col0, targets, indices);
        Assert.Equal(5.0f, threshold);
        Assert.True(gain > 0.3f, $"Expected positive Gini gain separating finite from NaN, got {gain}");

        var tree = new FastDecisionTree(maxDepth: 2, minSamplesSplit: 2);
        tree.Fit(matrix, targets);
        float[] preds = new float[totalCount];
        tree.Predict(matrix, preds);
        Assert.Equal(1.0f, Metrics.Accuracy(targets, preds));
    }
}
