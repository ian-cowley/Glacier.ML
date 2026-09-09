using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Glacier.ML.Core;

namespace Glacier.ML.Trees;

/// <summary>
/// Fast, zero-allocation decision tree classifier stored in a flat contiguous array of nodes.
/// </summary>
public sealed class FastDecisionTree : IPredictor
{
    private DecisionTreeNode[] _nodes = Array.Empty<DecisionTreeNode>();
    private readonly int _maxDepth;
    private readonly int _minSamplesSplit;

    public int MaxDepth => _maxDepth;
    public int MinSamplesSplit => _minSamplesSplit;
    public DecisionTreeNode[] Nodes => _nodes;

    public FastDecisionTree(int maxDepth = 6, int minSamplesSplit = 2)
    {
        _maxDepth = maxDepth;
        _minSamplesSplit = minSamplesSplit;
    }

    public void Fit(FeatureMatrix features, ReadOnlySpan<float> targets, int[]? featureIndices = null)
    {
        int numRows = features.Rows;
        int numCols = features.Columns;

        int[] activeFeatures = featureIndices ?? new int[numCols];
        if (featureIndices == null)
        {
            for (int c = 0; c < numCols; c++) activeFeatures[c] = c;
        }

        int[] sampleIndices = new int[numRows];
        for (int i = 0; i < numRows; i++) sampleIndices[i] = i;

        var nodeList = new List<DecisionTreeNode>();

        // Pre-extract columns for faster memory cache access
        float[][] colData = new float[numCols][];
        for (int c = 0; c < numCols; c++)
        {
            colData[c] = new float[numRows];
            features.CopyColumn(c, colData[c]);
        }

        BuildTreeRecursive(nodeList, colData, targets, sampleIndices, 0);
        _nodes = nodeList.ToArray();
    }

    private int BuildTreeRecursive(
        List<DecisionTreeNode> nodeList,
        float[][] colData,
        ReadOnlySpan<float> targets,
        int[] samples,
        int currentDepth)
    {
        int count = samples.Length;

        // Calculate majority class / mean target
        int posCount = 0;
        for (int i = 0; i < count; i++)
        {
            if (targets[samples[i]] > 0.5f) posCount++;
        }
        float leafVal = (posCount >= (count - posCount)) ? 1.0f : 0.0f;

        // Termination conditions
        if (currentDepth >= _maxDepth || count < _minSamplesSplit || posCount == 0 || posCount == count)
        {
            int leafIdx = nodeList.Count;
            nodeList.Add(DecisionTreeNode.CreateLeaf(leafVal));
            return leafIdx;
        }

        // Find best split across candidate features
        int bestFeature = -1;
        float bestThreshold = 0f;
        float bestGain = 0f;

        for (int f = 0; f < colData.Length; f++)
        {
            var (threshold, gain) = SplitFinder.FindBestSplitClassification(colData[f], targets, samples);
            if (gain > bestGain)
            {
                bestGain = gain;
                bestFeature = f;
                bestThreshold = threshold;
            }
        }

        if (bestGain <= 1e-6f || bestFeature == -1)
        {
            int leafIdx = nodeList.Count;
            nodeList.Add(DecisionTreeNode.CreateLeaf(leafVal));
            return leafIdx;
        }

        // Partition samples into left and right
        var leftSamples = new List<int>(count / 2);
        var rightSamples = new List<int>(count / 2);

        float[] bestCol = colData[bestFeature];
        for (int i = 0; i < count; i++)
        {
            int idx = samples[i];
            if (bestCol[idx] <= bestThreshold)
                leftSamples.Add(idx);
            else
                rightSamples.Add(idx);
        }

        if (leftSamples.Count == 0 || rightSamples.Count == 0)
        {
            int leafIdx = nodeList.Count;
            nodeList.Add(DecisionTreeNode.CreateLeaf(leafVal));
            return leafIdx;
        }

        // Reserve node slot in the flat array
        int nodeIdx = nodeList.Count;
        nodeList.Add(default); // Placeholder

        int leftChild = BuildTreeRecursive(nodeList, colData, targets, leftSamples.ToArray(), currentDepth + 1);
        int rightChild = BuildTreeRecursive(nodeList, colData, targets, rightSamples.ToArray(), currentDepth + 1);

        nodeList[nodeIdx] = DecisionTreeNode.CreateBranch(bestFeature, bestThreshold, leftChild, rightChild);
        return nodeIdx;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<float> predictions)
    {
        int rows = features.Rows;
        for (int r = 0; r < rows; r++)
        {
            predictions[r] = PredictRow(features.GetRow(r));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictRow(ReadOnlySpan<float> row)
    {
        if (_nodes.Length == 0) return 0f;

        int curr = 0;
        while (!_nodes[curr].IsLeaf)
        {
            int feat = _nodes[curr].FeatureIndex;
            float val = row[feat];
            curr = (val <= _nodes[curr].Threshold) ? _nodes[curr].LeftChild : _nodes[curr].RightChild;
        }

        return _nodes[curr].LeafValue;
    }
}
