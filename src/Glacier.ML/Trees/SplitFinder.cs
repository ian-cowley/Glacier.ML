using System;
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
    /// Evaluates the best split threshold for a given feature using Gini Impurity for binary classification.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static (float threshold, float gain) FindBestSplitClassification(
        ReadOnlySpan<float> featureValues,
        ReadOnlySpan<float> targets,
        ReadOnlySpan<int> sampleIndices)
    {
        int count = sampleIndices.Length;
        if (count < 2) return (0f, 0f);

        // Count total positives (class 1)
        int totalPos = 0;
        for (int i = 0; i < count; i++)
        {
            if (targets[sampleIndices[i]] > 0.5f)
                totalPos++;
        }
        int totalNeg = count - totalPos;

        if (totalPos == 0 || totalNeg == 0)
            return (0f, 0f); // Pure node

        float pPos = (float)totalPos / count;
        float pNeg = (float)totalNeg / count;
        float currentGini = 1.0f - (pPos * pPos + pNeg * pNeg);

        float bestGain = 0f;
        float bestThreshold = 0f;

        // Sample up to 32 candidate percentiles for splitting
        int step = Math.Max(1, count / 32);
        for (int s = 0; s < count; s += step)
        {
            float candidateThresh = featureValues[sampleIndices[s]];

            int leftPos = 0, leftNeg = 0;
            int rightPos = 0, rightNeg = 0;

            for (int i = 0; i < count; i++)
            {
                int idx = sampleIndices[i];
                bool isPos = targets[idx] > 0.5f;

                if (featureValues[idx] <= candidateThresh)
                {
                    if (isPos) leftPos++; else leftNeg++;
                }
                else
                {
                    if (isPos) rightPos++; else rightNeg++;
                }
            }

            int leftTotal = leftPos + leftNeg;
            int rightTotal = rightPos + rightNeg;

            if (leftTotal == 0 || rightTotal == 0)
                continue;

            float pLeftPos = (float)leftPos / leftTotal;
            float pLeftNeg = (float)leftNeg / leftTotal;
            float leftGini = 1.0f - (pLeftPos * pLeftPos + pLeftNeg * pLeftNeg);

            float pRightPos = (float)rightPos / rightTotal;
            float pRightNeg = (float)rightNeg / rightTotal;
            float rightGini = 1.0f - (pRightPos * pRightPos + pRightNeg * pRightNeg);

            float weightedGini = ((float)leftTotal / count) * leftGini + ((float)rightTotal / count) * rightGini;
            float gain = currentGini - weightedGini;

            if (gain > bestGain)
            {
                bestGain = gain;
                bestThreshold = candidateThresh;
            }
        }

        return (bestThreshold, bestGain);
    }
}
