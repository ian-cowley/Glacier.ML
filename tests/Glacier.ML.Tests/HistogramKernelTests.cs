using System;
using Glacier.ML.Trees;
using Xunit;

namespace Glacier.ML.Tests;

public class HistogramKernelTests
{
    [Fact]
    public void DiscretizeFeature_BinsCorrectly()
    {
        ReadOnlySpan<float> values = stackalloc float[] { 0.1f, 0.5f, 1.2f, 2.5f, 3.8f, 5.0f };
        ReadOnlySpan<float> thresholds = stackalloc float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        Span<byte> bins = stackalloc byte[values.Length];

        HistogramKernels.DiscretizeFeature(values, thresholds, bins);

        Assert.Equal(0, bins[0]); // 0.1 <= 1.0 -> bin 0
        Assert.Equal(0, bins[1]); // 0.5 <= 1.0 -> bin 0
        Assert.Equal(1, bins[2]); // 1.2 <= 2.0 -> bin 1
        Assert.Equal(2, bins[3]); // 2.5 <= 3.0 -> bin 2
        Assert.Equal(3, bins[4]); // 3.8 <= 4.0 -> bin 3
        Assert.Equal(4, bins[5]); // 5.0 > 4.0 -> bin 4
    }

    [Fact]
    public void BuildHistogram_MatchesScalarAccumulation()
    {
        const int count = 1000;
        byte[] bins = new byte[count];
        float[] grads = new float[count];
        float[] expected = new float[256];

        var rng = new Random(42);
        for (int i = 0; i < count; i++)
        {
            bins[i] = (byte)rng.Next(0, 256);
            grads[i] = (float)rng.NextDouble();
            expected[bins[i]] += grads[i];
        }

        float[] actual = new float[256];
        HistogramKernels.BuildHistogram(bins, grads, actual);

        for (int b = 0; b < 256; b++)
        {
            Assert.Equal(expected[b], actual[b], precision: 4);
        }
    }

    [Fact]
    public void BuildHistogramWithCounts_MatchesScalar()
    {
        const int count = 500;
        byte[] bins = new byte[count];
        float[] grads = new float[count];
        float[] expectedGrads = new float[256];
        int[] expectedCounts = new int[256];

        var rng = new Random(123);
        for (int i = 0; i < count; i++)
        {
            bins[i] = (byte)rng.Next(0, 10);
            grads[i] = (float)(rng.NextDouble() * 10 - 5);
            expectedGrads[bins[i]] += grads[i];
            expectedCounts[bins[i]]++;
        }

        float[] actualGrads = new float[256];
        int[] actualCounts = new int[256];
        HistogramKernels.BuildHistogram(bins, grads, actualGrads, actualCounts);

        for (int b = 0; b < 256; b++)
        {
            Assert.Equal(expectedGrads[b], actualGrads[b], precision: 4);
            Assert.Equal(expectedCounts[b], actualCounts[b]);
        }
    }
}
