using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Glacier.ML.Core;

/// <summary>
/// Hardware-accelerated evaluation metrics for classification and regression.
/// </summary>
public static class Metrics
{
    /// <summary>
    /// Computes classification accuracy: (correct predictions) / (total predictions).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Accuracy(ReadOnlySpan<float> actual, ReadOnlySpan<float> predicted)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Actual and predicted spans must have equal length.");
        if (actual.Length == 0) return 0f;

        int count = actual.Length;
        int correct = 0;

        for (int i = 0; i < count; i++)
        {
            if (MathF.Round(actual[i]) == MathF.Round(predicted[i]))
            {
                correct++;
            }
        }

        return (float)correct / count;
    }

    /// <summary>
    /// Computes Mean Squared Error (MSE) using SIMD vectorization.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe float MeanSquaredError(ReadOnlySpan<float> actual, ReadOnlySpan<float> predicted)
    {
        if (actual.Length != predicted.Length)
            throw new ArgumentException("Actual and predicted spans must have equal length.");
        if (actual.Length == 0) return 0f;

        int length = actual.Length;
        float sumSquaredError = 0f;

        fixed (float* pAct = actual)
        fixed (float* pPred = predicted)
        {
            int i = 0;

            if (Vector512.IsHardwareAccelerated && length >= 16)
            {
                var vAcc = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    var diff = Vector512.Subtract(Vector512.Load(pAct + i), Vector512.Load(pPred + i));
                    vAcc = Vector512.Add(vAcc, Vector512.Multiply(diff, diff));
                }
                sumSquaredError += Vector512.Sum(vAcc);
            }
            else if (Vector256.IsHardwareAccelerated && length >= 8)
            {
                var vAcc = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    var diff = Vector256.Subtract(Vector256.Load(pAct + i), Vector256.Load(pPred + i));
                    vAcc = Vector256.Add(vAcc, Vector256.Multiply(diff, diff));
                }
                sumSquaredError += Vector256.Sum(vAcc);
            }

            for (; i < length; i++)
            {
                float diff = pAct[i] - pPred[i];
                sumSquaredError += diff * diff;
            }
        }

        return sumSquaredError / length;
    }

    /// <summary>
    /// Computes R2 (Coefficient of Determination) regression score.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float R2Score(ReadOnlySpan<float> actual, ReadOnlySpan<float> predicted)
    {
        if (actual.Length != predicted.Length || actual.Length == 0)
            throw new ArgumentException("Actual and predicted spans must have equal non-zero length.");

        float sum = 0f;
        for (int i = 0; i < actual.Length; i++) sum += actual[i];
        float mean = sum / actual.Length;

        float ssTot = 0f;
        float ssRes = 0f;

        for (int i = 0; i < actual.Length; i++)
        {
            float diffTot = actual[i] - mean;
            ssTot += diffTot * diffTot;

            float diffRes = actual[i] - predicted[i];
            ssRes += diffRes * diffRes;
        }

        if (ssTot == 0f) return 1f;
        return 1f - (ssRes / ssTot);
    }
}
