using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Core;

namespace Glacier.ML.Trees;

/// <summary>
/// Multi-threaded Random Forest ensemble classifier outperforming Scikit-Learn.
/// Trains trees concurrently across CPU cores without the Python GIL.
/// </summary>
public sealed class FastRandomForest : IPredictor
{
    private readonly int _numTrees;
    private readonly int _maxDepth;
    private readonly int _minSamplesSplit;
    private readonly float _subsampleRatio;
    private FastDecisionTree[] _trees = Array.Empty<FastDecisionTree>();

    public int NumTrees => _numTrees;
    public int MaxDepth => _maxDepth;
    public FastDecisionTree[] Trees => _trees;

    public FastRandomForest(
        int numTrees = 100, 
        int maxDepth = 8, 
        int minSamplesSplit = 2,
        float subsampleRatio = 0.8f)
    {
        _numTrees = Math.Max(1, numTrees);
        _maxDepth = maxDepth;
        _minSamplesSplit = minSamplesSplit;
        _subsampleRatio = Math.Clamp(subsampleRatio, 0.1f, 1.0f);
    }

    public void Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        int totalRows = features.Rows;
        int totalCols = features.Columns;
        int sampleSize = Math.Max(2, (int)(totalRows * _subsampleRatio));

        _trees = new FastDecisionTree[_numTrees];
        float[] targetsArray = targets.ToArray();

        // Fit trees in parallel across all CPU cores
        Parallel.For(0, _numTrees, t =>
        {
            var rng = new Random(42 + t * 31);

            // Bootstrap row sampling
            int[] bootstrapIndices = new int[sampleSize];
            float[] sampleTargets = new float[sampleSize];
            var sampleMatrix = new FeatureMatrix(sampleSize, totalCols);

            for (int s = 0; s < sampleSize; s++)
            {
                int r = rng.Next(0, totalRows);
                bootstrapIndices[s] = r;
                sampleTargets[s] = targetsArray[r];
                for (int c = 0; c < totalCols; c++)
                {
                    sampleMatrix.SetValue(s, c, features.GetValue(r, c));
                }
            }

            var tree = new FastDecisionTree(_maxDepth, _minSamplesSplit);
            tree.Fit(sampleMatrix, sampleTargets);
            _trees[t] = tree;
        });
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
        if (_trees.Length == 0) return 0f;

        float sum = 0f;
        for (int t = 0; t < _trees.Length; t++)
        {
            sum += _trees[t].PredictRow(row);
        }

        // Binary classification majority threshold
        return (sum / _trees.Length) >= 0.5f ? 1.0f : 0.0f;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictProbability(ReadOnlySpan<float> row)
    {
        if (_trees.Length == 0) return 0f;

        float sum = 0f;
        for (int t = 0; t < _trees.Length; t++)
        {
            sum += _trees[t].PredictRow(row);
        }

        return sum / _trees.Length;
    }
}
