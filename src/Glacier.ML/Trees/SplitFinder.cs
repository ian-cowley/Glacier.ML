using System;
using System.Buffers;
using System.Runtime.CompilerServices;

namespace Glacier.ML.Trees;

public readonly struct SplitResult
{
    public readonly int FeatureIndex;
    public readonly float Threshold;
    public readonly float Gain;
    public readonly bool Found;

    public SplitResult(int featureIndex, float threshold, float gain)
    {
        FeatureIndex = featureIndex;
        Threshold = threshold;
        Gain = gain;
        Found = true;
    }

    public SplitResult(int featureIndex, float threshold, float gain, bool found)
    {
        FeatureIndex = featureIndex;
        Threshold = threshold;
        Gain = gain;
        Found = found;
    }

    public static readonly SplitResult None = new(-1, 0f, 0f, false);
}

public static class SplitFinder
{
    /// <summary>
    /// Evaluates the best split threshold for a given feature using Gini Impurity for binary classification
    /// (<c>target &gt; 0.5f</c> is treated as positive class 1, otherwise class 0).
    /// Uses exact scan for small nodes (count &lt;= 32) and SIMD 4-way histogram quantile splitting for larger nodes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static (float threshold, float gain) FindBestSplitClassification(
        ReadOnlySpan<float> featureValues,
        ReadOnlySpan<float> targets,
        ReadOnlySpan<int> sampleIndices)
    {
        int count = sampleIndices.Length;
        if (count < 2) return (0f, 0f);

        int totalPos = 0;
        float minVal = float.PositiveInfinity;
        float maxVal = float.NegativeInfinity;
        int finiteCount = 0;

        for (int i = 0; i < count; i++)
        {
            int idx = sampleIndices[i];
            if (targets[idx] > 0.5f)
                totalPos++;

            float v = featureValues[idx];
            if (!float.IsNaN(v))
            {
                finiteCount++;
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        }

        int totalNeg = count - totalPos;
        if (totalPos == 0 || totalNeg == 0 || finiteCount == 0 || (maxVal <= minVal && finiteCount == count))
            return (0f, 0f);

        float pPos = (float)totalPos / count;
        float pNeg = (float)totalNeg / count;
        float currentGini = 1.0f - (pPos * pPos + pNeg * pNeg);

        if (count <= 32)
        {
            return FindBestSplitExactSmall(featureValues, targets, sampleIndices, count, totalPos, currentGini);
        }

        return FindBestSplitHistogram(featureValues, targets, sampleIndices, count, totalPos, finiteCount, minVal, maxVal, currentGini);
    }

    private static (float threshold, float gain) FindBestSplitExactSmall(
        ReadOnlySpan<float> featureValues,
        ReadOnlySpan<float> targets,
        ReadOnlySpan<int> sampleIndices,
        int count,
        int totalPos,
        float currentGini)
    {
        Span<float> vals = stackalloc float[count];
        Span<byte> pos = stackalloc byte[count];

        int finiteCount = 0;
        int nanTail = count;
        for (int i = 0; i < count; i++)
        {
            int idx = sampleIndices[i];
            float v = featureValues[idx];
            byte isPos = targets[idx] > 0.5f ? (byte)1 : (byte)0;
            if (float.IsNaN(v))
            {
                nanTail--;
                vals[nanTail] = v;
                pos[nanTail] = isPos;
            }
            else
            {
                vals[finiteCount] = v;
                pos[finiteCount] = isPos;
                finiteCount++;
            }
        }

        // Insertion sort finite values ascending
        for (int i = 1; i < finiteCount; i++)
        {
            float keyV = vals[i];
            byte keyP = pos[i];
            int j = i - 1;
            while (j >= 0 && vals[j] > keyV)
            {
                vals[j + 1] = vals[j];
                pos[j + 1] = pos[j];
                j--;
            }
            vals[j + 1] = keyV;
            pos[j + 1] = keyP;
        }

        float bestGain = 0f;
        float bestThreshold = 0f;
        int leftPos = 0;

        for (int i = 0; i < finiteCount; i++)
        {
            leftPos += pos[i];
            int leftTotal = i + 1;
            if (leftTotal == count)
                break;

            if (i + 1 < finiteCount && vals[i] == vals[i + 1])
                continue;

            int rightTotal = count - leftTotal;
            int rightPos = totalPos - leftPos;

            float gain = ComputeGiniGain(currentGini, count, leftPos, leftTotal, rightPos, rightTotal);
            if (gain > bestGain)
            {
                bestGain = gain;
                bestThreshold = vals[i];
            }
        }

        return (bestThreshold, bestGain);
    }

    private static (float threshold, float gain) FindBestSplitHistogram(
        ReadOnlySpan<float> featureValues,
        ReadOnlySpan<float> targets,
        ReadOnlySpan<int> sampleIndices,
        int count,
        int totalPos,
        int finiteCount,
        float minVal,
        float maxVal,
        float currentGini)
    {
        const int MaxSampleCount = 256;
        Span<float> sampleVals = stackalloc float[MaxSampleCount];
        int sampled = 0;
        for (int s = 0; s < MaxSampleCount; s++)
        {
            int idx = sampleIndices[(int)((long)s * (count - 1) / (MaxSampleCount - 1))];
            float v = featureValues[idx];
            if (!float.IsNaN(v))
                sampleVals[sampled++] = v;
        }

        Span<float> thresholds = stackalloc float[255];
        int numThresholds = 0;

        if (sampled > 0)
        {
            MemoryExtensions.Sort(sampleVals[..sampled]);
            for (int i = 0; i < sampled && numThresholds < 255; i++)
            {
                float v = sampleVals[i];
                if (v >= maxVal)
                    break;
                if (numThresholds == 0 || v > thresholds[numThresholds - 1])
                {
                    thresholds[numThresholds++] = v;
                }
            }
        }

        if (numThresholds == 0)
        {
            thresholds[0] = minVal;
            numThresholds = 1;
        }

        // When NaN values are present (finiteCount < count), include maxVal as the final threshold
        // so splitting all finite values (<= maxVal) LEFT vs NaN RIGHT (bin numThresholds) is evaluated.
        if (finiteCount < count && thresholds[numThresholds - 1] < maxVal)
        {
            if (numThresholds < 255)
            {
                thresholds[numThresholds++] = maxVal;
            }
            else
            {
                thresholds[254] = maxVal;
            }
        }

        ReadOnlySpan<float> activeThresholds = thresholds[..numThresholds];

        float[]? rentedValues = null;
        float[]? rentedGrads = null;
        byte[]? rentedBins = null;

        try
        {
            Span<float> valuesBuf = count <= 1024
                ? stackalloc float[count]
                : (rentedValues = ArrayPool<float>.Shared.Rent(count)).AsSpan(0, count);
            Span<float> gradsBuf = count <= 1024
                ? stackalloc float[count]
                : (rentedGrads = ArrayPool<float>.Shared.Rent(count)).AsSpan(0, count);
            Span<byte> binnedBuf = count <= 1024
                ? stackalloc byte[count]
                : (rentedBins = ArrayPool<byte>.Shared.Rent(count)).AsSpan(0, count);

            for (int i = 0; i < count; i++)
            {
                int idx = sampleIndices[i];
                valuesBuf[i] = featureValues[idx];
                gradsBuf[i] = targets[idx] > 0.5f ? 1.0f : 0.0f;
            }

            HistogramKernels.DiscretizeFeature(valuesBuf, activeThresholds, binnedBuf);

            Span<float> histPos = stackalloc float[256];
            Span<int> histCounts = stackalloc int[256];
            HistogramKernels.BuildHistogram(binnedBuf, gradsBuf, histPos, histCounts);

            float bestGain = 0f;
            float bestThreshold = 0f;
            int leftTotal = 0;
            int leftPos = 0;

            for (int b = 0; b < numThresholds; b++)
            {
                int binCount = histCounts[b];
                leftTotal += binCount;
                leftPos += Math.Clamp((int)MathF.Round(histPos[b]), 0, binCount);

                int rightTotal = count - leftTotal;
                if (leftTotal == 0 || rightTotal == 0)
                    continue;

                int rightPos = totalPos - leftPos;
                float gain = ComputeGiniGain(currentGini, count, leftPos, leftTotal, rightPos, rightTotal);
                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestThreshold = activeThresholds[b];
                }
            }

            return (bestThreshold, bestGain);
        }
        finally
        {
            if (rentedValues != null) ArrayPool<float>.Shared.Return(rentedValues);
            if (rentedGrads != null) ArrayPool<float>.Shared.Return(rentedGrads);
            if (rentedBins != null) ArrayPool<byte>.Shared.Return(rentedBins);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeGiniGain(
        float currentGini,
        int totalCount,
        int leftPos,
        int leftTotal,
        int rightPos,
        int rightTotal)
    {
        float invLeft = 1.0f / leftTotal;
        float pLeftPos = leftPos * invLeft;
        float pLeftNeg = 1.0f - pLeftPos;
        float leftGini = 1.0f - (pLeftPos * pLeftPos + pLeftNeg * pLeftNeg);

        float invRight = 1.0f / rightTotal;
        float pRightPos = rightPos * invRight;
        float pRightNeg = 1.0f - pRightPos;
        float rightGini = 1.0f - (pRightPos * pRightPos + pRightNeg * pRightNeg);

        float invTotal = 1.0f / totalCount;
        float weightedGini = (leftTotal * invTotal) * leftGini + (rightTotal * invTotal) * rightGini;
        return currentGini - weightedGini;
    }
}
