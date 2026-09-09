using System;
using BenchmarkDotNet.Attributes;
using Glacier.ML.Core;
using Glacier.ML.Trees;

namespace Glacier.ML.Benchmarks;

[MemoryDiagnoser]
public class RandomForestBenchmarks
{
    private FeatureMatrix _matrix = null!;
    private float[] _targets = null!;
    private FastRandomForest _forest = null!;
    private float[] _singleSample = null!;

    [GlobalSetup]
    public void Setup()
    {
        const int n = 50_000;
        const int cols = 8;
        _matrix = new FeatureMatrix(n, cols);
        _targets = new float[n];
        _singleSample = new float[cols];

        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            float sum = 0f;
            for (int c = 0; c < cols; c++)
            {
                float v = (float)rng.NextDouble() * 10f;
                _matrix.SetValue(i, c, v);
                sum += v;
            }
            _targets[i] = (sum > cols * 5f) ? 1.0f : 0.0f;
        }

        for (int c = 0; c < cols; c++) _singleSample[c] = 5.5f;

        _forest = new FastRandomForest(numTrees: 20, maxDepth: 6);
        _forest.Fit(_matrix, _targets);
    }

    [Benchmark]
    public void TrainRandomForest_20Trees()
    {
        var forest = new FastRandomForest(numTrees: 20, maxDepth: 6);
        forest.Fit(_matrix, _targets);
    }

    [Benchmark]
    public float PredictSingleRow_ZeroAlloc()
    {
        return _forest.PredictRow(_singleSample);
    }
}
