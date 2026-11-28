using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Glacier.ML.Core;

namespace Glacier.ML.Trees;

/// <summary>
/// Fast, zero-allocation binary decision tree classifier (<c>target &gt; 0.5f</c> -&gt; 1.0f, else 0.0f)
/// stored in a flat contiguous array of nodes.
/// </summary>
public sealed class FastDecisionTree : IPredictor, IEstimator<FastDecisionTree>
{
    private DecisionTreeNode[] _nodes = Array.Empty<DecisionTreeNode>();
    private readonly int _maxDepth;
    private readonly int _minSamplesSplit;
    private int _nFeatures;

    public int MaxDepth => _maxDepth;
    public int MinSamplesSplit => _minSamplesSplit;
    public int NFeatures => _nFeatures;
    public DecisionTreeNode[] Nodes => _nodes;

    public FastDecisionTree(int maxDepth = 6, int minSamplesSplit = 2)
    {
        if (maxDepth <= 0) throw new ArgumentOutOfRangeException(nameof(maxDepth), "MaxDepth must be positive.");
        if (minSamplesSplit < 2) throw new ArgumentOutOfRangeException(nameof(minSamplesSplit), "MinSamplesSplit must be at least 2.");
        _maxDepth = maxDepth;
        _minSamplesSplit = minSamplesSplit;
    }

    FastDecisionTree IEstimator<FastDecisionTree>.Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        Fit(features, targets);
        return this;
    }

    public void Fit(
        FeatureMatrix features, 
        ReadOnlySpan<float> targets, 
        int[]? featureIndices = null, 
        int[]? sampleIndices = null)
    {
        ArgumentNullException.ThrowIfNull(features);
        int numRows = features.Rows;
        int numCols = features.Columns;
        if (numRows <= 0)
            throw new ArgumentException("FeatureMatrix must contain at least one row.", nameof(features));
        if (targets.Length < numRows)
            throw new ArgumentException("Targets length must match or exceed feature rows.", nameof(targets));
        if (sampleIndices != null && sampleIndices.Length == 0)
            throw new ArgumentException("Sample indices cannot be empty.", nameof(sampleIndices));

        _nFeatures = numCols;

        int[] activeFeatures = featureIndices ?? new int[numCols];
        if (featureIndices == null)
        {
            for (int c = 0; c < numCols; c++) activeFeatures[c] = c;
        }

        int[] samples;
        if (sampleIndices != null)
        {
            samples = (int[])sampleIndices.Clone();
        }
        else
        {
            samples = new int[numRows];
            for (int i = 0; i < numRows; i++) samples[i] = i;
        }

        var nodeList = new List<DecisionTreeNode>(128);

        if (features.IsColumnar)
        {
            // Zero-copy path: directly pass indexed column spans without allocating colData
            BuildTreeInPlaceColumnar(nodeList, features, targets, activeFeatures, samples, 0, samples.Length, 0);
        }
        else
        {
            // Row-major path: pre-extract columns for faster memory cache access
            float[][] colData = new float[numCols][];
            for (int c = 0; c < numCols; c++)
            {
                colData[c] = new float[numRows];
                features.CopyColumn(c, colData[c]);
            }

            BuildTreeInPlace(nodeList, colData, targets, activeFeatures, samples, 0, samples.Length, 0);
        }

        _nodes = nodeList.ToArray();
    }

    private int BuildTreeInPlaceColumnar(
        List<DecisionTreeNode> nodeList,
        FeatureMatrix features,
        ReadOnlySpan<float> targets,
        int[] activeFeatures,
        int[] samples,
        int offset,
        int count,
        int currentDepth)
    {
        ReadOnlySpan<int> sampleSlice = samples.AsSpan(offset, count);

        // Calculate majority class / mean target
        int posCount = 0;
        for (int i = 0; i < count; i++)
        {
            if (targets[sampleSlice[i]] > 0.5f) posCount++;
        }
        float leafVal = (posCount >= (count - posCount)) ? 1.0f : 0.0f;

        // Termination conditions
        if (currentDepth >= _maxDepth || count < _minSamplesSplit || posCount == 0 || posCount == count)
        {
            int leafIdx = nodeList.Count;
            nodeList.Add(DecisionTreeNode.CreateLeaf(leafVal));
            return leafIdx;
        }

        // Find best split across candidate features using zero-copy column spans
        int bestFeature = -1;
        float bestThreshold = 0f;
        float bestGain = 0f;

        for (int fi = 0; fi < activeFeatures.Length; fi++)
        {
            int f = activeFeatures[fi];
            var (threshold, gain) = SplitFinder.FindBestSplitClassification(features.GetColumn(f), targets, sampleSlice);
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

        // ZERO ALLOCATION: In-place two-pointer Hoare partition within samples[offset .. offset + count]
        ReadOnlySpan<float> bestCol = features.GetColumn(bestFeature);
        int left = offset;
        int right = offset + count - 1;

        while (left <= right)
        {
            if (bestCol[samples[left]] <= bestThreshold)
            {
                left++;
            }
            else
            {
                (samples[left], samples[right]) = (samples[right], samples[left]);
                right--;
            }
        }

        int leftCount = left - offset;
        int rightCount = count - leftCount;

        if (leftCount == 0 || rightCount == 0)
        {
            int leafIdx = nodeList.Count;
            nodeList.Add(DecisionTreeNode.CreateLeaf(leafVal));
            return leafIdx;
        }

        // Reserve node slot in the flat array
        int nodeIdx = nodeList.Count;
        nodeList.Add(default); // Placeholder

        int leftChild = BuildTreeInPlaceColumnar(nodeList, features, targets, activeFeatures, samples, offset, leftCount, currentDepth + 1);
        int rightChild = BuildTreeInPlaceColumnar(nodeList, features, targets, activeFeatures, samples, left, rightCount, currentDepth + 1);

        nodeList[nodeIdx] = DecisionTreeNode.CreateBranch(bestFeature, bestThreshold, leftChild, rightChild);
        return nodeIdx;
    }

    private int BuildTreeInPlace(
        List<DecisionTreeNode> nodeList,
        float[][] colData,
        ReadOnlySpan<float> targets,
        int[] activeFeatures,
        int[] samples,
        int offset,
        int count,
        int currentDepth)
    {
        ReadOnlySpan<int> sampleSlice = samples.AsSpan(offset, count);

        // Calculate majority class / mean target
        int posCount = 0;
        for (int i = 0; i < count; i++)
        {
            if (targets[sampleSlice[i]] > 0.5f) posCount++;
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

        for (int fi = 0; fi < activeFeatures.Length; fi++)
        {
            int f = activeFeatures[fi];
            var (threshold, gain) = SplitFinder.FindBestSplitClassification(colData[f], targets, sampleSlice);
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

        // ZERO ALLOCATION: In-place two-pointer Hoare partition within samples[offset .. offset + count]
        float[] bestCol = colData[bestFeature];
        int left = offset;
        int right = offset + count - 1;

        while (left <= right)
        {
            if (bestCol[samples[left]] <= bestThreshold)
            {
                left++;
            }
            else
            {
                (samples[left], samples[right]) = (samples[right], samples[left]);
                right--;
            }
        }

        int leftCount = left - offset;
        int rightCount = count - leftCount;

        if (leftCount == 0 || rightCount == 0)
        {
            int leafIdx = nodeList.Count;
            nodeList.Add(DecisionTreeNode.CreateLeaf(leafVal));
            return leafIdx;
        }

        // Reserve node slot in the flat array
        int nodeIdx = nodeList.Count;
        nodeList.Add(default); // Placeholder

        int leftChild = BuildTreeInPlace(nodeList, colData, targets, activeFeatures, samples, offset, leftCount, currentDepth + 1);
        int rightChild = BuildTreeInPlace(nodeList, colData, targets, activeFeatures, samples, left, rightCount, currentDepth + 1);

        nodeList[nodeIdx] = DecisionTreeNode.CreateBranch(bestFeature, bestThreshold, leftChild, rightChild);
        return nodeIdx;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<float> predictions)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (_nodes.Length == 0 || _nFeatures <= 0)
            throw new InvalidOperationException("FastDecisionTree must be fit before calling Predict.");
        int rows = features.Rows;
        if (features.Columns != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {features.Columns}.", nameof(features));
        if (predictions.Length < rows)
            throw new ArgumentException("Predictions span is smaller than feature rows.", nameof(predictions));

        if (features.IsColumnar)
        {
            for (int r = 0; r < rows; r++)
            {
                predictions[r] = PredictRowColumnar(features, r);
            }
        }
        else
        {
            ReadOnlySpan<float> data = features.Span;
            int cols = _nFeatures;
            for (int r = 0; r < rows; r++)
            {
                predictions[r] = PredictRow(data.Slice(r * cols, cols));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictRow(ReadOnlySpan<float> row)
    {
        if (_nodes.Length == 0 || _nFeatures <= 0)
            throw new InvalidOperationException("FastDecisionTree must be fit before calling PredictRow.");
        if (row.Length != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {row.Length}.", nameof(row));

        int curr = 0;
        while (!_nodes[curr].IsLeaf)
        {
            int feat = _nodes[curr].FeatureIndex;
            float val = row[feat];
            curr = (val <= _nodes[curr].Threshold) ? _nodes[curr].LeftChild : _nodes[curr].RightChild;
        }

        return _nodes[curr].LeafValue;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictRowColumnar(FeatureMatrix features, int rowIndex)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (_nodes.Length == 0 || _nFeatures <= 0)
            throw new InvalidOperationException("FastDecisionTree must be fit before calling PredictRowColumnar.");
        if (features.Columns != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {features.Columns}.", nameof(features));

        int curr = 0;
        while (!_nodes[curr].IsLeaf)
        {
            int feat = _nodes[curr].FeatureIndex;
            float val = features.GetColumn(feat)[rowIndex];
            curr = (val <= _nodes[curr].Threshold) ? _nodes[curr].LeftChild : _nodes[curr].RightChild;
        }

        return _nodes[curr].LeafValue;
    }
}
