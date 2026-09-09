using System;
using BenchmarkDotNet.Attributes;
using Glacier.ML.Trees;

namespace Glacier.ML.Benchmarks;

[MemoryDiagnoser]
public class HistogramBenchmarks
{
    private byte[] _binnedFeature = null!;
    private float[] _gradients = null!;
    private float[] _outputHist = null!;

    [Params(100_000, 1_000_000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        _binnedFeature = new byte[N];
        _gradients = new float[N];
        _outputHist = new float[256];

        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            _binnedFeature[i] = (byte)rng.Next(0, 256);
            _gradients[i] = (float)rng.NextDouble();
        }
    }

    [Benchmark(Baseline = true)]
    public void ScalarHistogram()
    {
        Array.Clear(_outputHist);
        for (int i = 0; i < _binnedFeature.Length; i++)
        {
            _outputHist[_binnedFeature[i]] += _gradients[i];
        }
    }

    [Benchmark]
    public void VectorizedHistogram4Way()
    {
        HistogramKernels.BuildHistogram(_binnedFeature, _gradients, _outputHist);
    }
}
