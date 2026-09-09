using System;
using BenchmarkDotNet.Attributes;
using Glacier.ML.Clustering;
using Glacier.ML.Core;

namespace Glacier.ML.Benchmarks;

[MemoryDiagnoser]
public class KMeansBenchmarks
{
    private FeatureMatrix _matrix = null!;
    private KMeans _kmeans = null!;
    private float[] _queryRow = null!;

    [Params(50_000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        const int cols = 8;
        _matrix = new FeatureMatrix(N, cols);
        _queryRow = new float[cols];

        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            for (int c = 0; c < cols; c++)
            {
                _matrix.SetValue(i, c, (float)rng.NextDouble() * 100f);
            }
        }

        for (int c = 0; c < cols; c++) _queryRow[c] = 50f;

        _kmeans = new KMeans(k: 8, maxIterations: 20);
        _kmeans.Fit(_matrix);
    }

    [Benchmark]
    public void TrainKMeans_8Clusters()
    {
        var kmeans = new KMeans(k: 8, maxIterations: 10);
        kmeans.Fit(_matrix);
    }

    [Benchmark]
    public int PredictSingleSample()
    {
        return _kmeans.PredictRow(_queryRow);
    }
}
