using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Glacier.ML.Clustering;

/// <summary>
/// Hardware-accelerated distance kernels for k-means clustering.
/// Saturated memory-bandwidth centroid distance evaluations.
/// </summary>
public static unsafe class KMeansKernels
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int FindNearestCentroid(
        ReadOnlySpan<float> sample, 
        ReadOnlySpan<float> centroids, 
        int numCentroids, 
        int dimensions)
    {
        fixed (float* pSample = sample)
        fixed (float* pCent = centroids)
        {
            return FindNearestCentroid(pSample, pCent, numCentroids, dimensions);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int FindNearestCentroid(
        float* pSample, 
        int dimensions, // not used directly in signature order, keeping signature
        float* pCentroids, 
        int numCentroids) => FindNearestCentroid(pSample, pCentroids, numCentroids, dimensions);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int FindNearestCentroidDim2(float* pSample, float* pCentroids, int numCentroids)
    {
        float s0 = pSample[0];
        float s1 = pSample[1];
        int bestCluster = 0;
        float minDistance = float.MaxValue;

        for (int k = 0; k < numCentroids; k++)
        {
            float* pC = pCentroids + (k * 2);
            float diff0 = s0 - pC[0];
            float diff1 = s1 - pC[1];
            float dist = diff0 * diff0 + diff1 * diff1;
            if (dist < minDistance)
            {
                minDistance = dist;
                bestCluster = k;
            }
        }
        return bestCluster;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int FindNearestCentroidDim3(float* pSample, float* pCentroids, int numCentroids)
    {
        float s0 = pSample[0];
        float s1 = pSample[1];
        float s2 = pSample[2];
        int bestCluster = 0;
        float minDistance = float.MaxValue;

        for (int k = 0; k < numCentroids; k++)
        {
            float* pC = pCentroids + (k * 3);
            float diff0 = s0 - pC[0];
            float diff1 = s1 - pC[1];
            float diff2 = s2 - pC[2];
            float dist = diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
            if (dist < minDistance)
            {
                minDistance = dist;
                bestCluster = k;
            }
        }
        return bestCluster;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int FindNearestCentroidDim4(float* pSample, float* pCentroids, int numCentroids)
    {
        int bestCluster = 0;
        float minDistance = float.MaxValue;

        if (Vector128.IsHardwareAccelerated)
        {
            var vSample = Vector128.Load(pSample);
            for (int k = 0; k < numCentroids; k++)
            {
                var vC = Vector128.Load(pCentroids + (k * 4));
                var diff = Vector128.Subtract(vSample, vC);
                float dist = Vector128.Dot(diff, diff);
                if (dist < minDistance)
                {
                    minDistance = dist;
                    bestCluster = k;
                }
            }
            return bestCluster;
        }

        float s0 = pSample[0], s1 = pSample[1], s2 = pSample[2], s3 = pSample[3];
        for (int k = 0; k < numCentroids; k++)
        {
            float* pC = pCentroids + (k * 4);
            float diff0 = s0 - pC[0], diff1 = s1 - pC[1], diff2 = s2 - pC[2], diff3 = s3 - pC[3];
            float dist = diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            if (dist < minDistance)
            {
                minDistance = dist;
                bestCluster = k;
            }
        }
        return bestCluster;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int FindNearestCentroid(
        float* pSample, 
        float* pCentroids, 
        int numCentroids, 
        int dimensions)
    {
        if (dimensions == 2) return FindNearestCentroidDim2(pSample, pCentroids, numCentroids);
        if (dimensions == 3) return FindNearestCentroidDim3(pSample, pCentroids, numCentroids);
        if (dimensions == 4) return FindNearestCentroidDim4(pSample, pCentroids, numCentroids);

        int bestCluster = 0;
        float minDistance = float.MaxValue;

        for (int k = 0; k < numCentroids; k++)
        {
            float* pC = pCentroids + (k * dimensions);
            float dist = 0f;
            int d = 0;

            if (Vector512.IsHardwareAccelerated && dimensions >= 16)
            {
                var acc = Vector512<float>.Zero;
                for (; d <= dimensions - 16; d += 16)
                {
                    var diff = Vector512.Subtract(Vector512.Load(pSample + d), Vector512.Load(pC + d));
                    acc = Vector512.MultiplyAddEstimate(diff, diff, acc);
                }
                dist = Vector512.Sum(acc);
            }
            else if (Vector256.IsHardwareAccelerated && dimensions >= 8)
            {
                var acc = Vector256<float>.Zero;
                for (; d <= dimensions - 8; d += 8)
                {
                    var diff = Vector256.Subtract(Vector256.Load(pSample + d), Vector256.Load(pC + d));
                    acc = Vector256.MultiplyAddEstimate(diff, diff, acc);
                }
                dist = Vector256.Sum(acc);
            }
            else if (Vector128.IsHardwareAccelerated && dimensions >= 4)
            {
                var acc = Vector128<float>.Zero;
                for (; d <= dimensions - 4; d += 4)
                {
                    var diff = Vector128.Subtract(Vector128.Load(pSample + d), Vector128.Load(pC + d));
                    acc = Vector128.MultiplyAddEstimate(diff, diff, acc);
                }
                dist = Vector128.Sum(acc);
            }

            // Remainder scalar loop
            for (; d < dimensions; d++)
            {
                float diff = pSample[d] - pC[d];
                dist += diff * diff;
            }

            if (dist < minDistance)
            {
                minDistance = dist;
                bestCluster = k;
            }
        }

        return bestCluster;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float EuclideanDistanceSquared(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int length = a.Length;
        float sum = 0f;

        fixed (float* pA = a)
        fixed (float* pB = b)
        {
            int i = 0;
            if (Vector512.IsHardwareAccelerated && length >= 16)
            {
                var acc = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    var diff = Vector512.Subtract(Vector512.Load(pA + i), Vector512.Load(pB + i));
                    acc = Vector512.MultiplyAddEstimate(diff, diff, acc);
                }
                sum += Vector512.Sum(acc);
            }
            else if (Vector256.IsHardwareAccelerated && length >= 8)
            {
                var acc = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    var diff = Vector256.Subtract(Vector256.Load(pA + i), Vector256.Load(pB + i));
                    acc = Vector256.MultiplyAddEstimate(diff, diff, acc);
                }
                sum += Vector256.Sum(acc);
            }
            else if (Vector128.IsHardwareAccelerated && length >= 4)
            {
                var acc = Vector128<float>.Zero;
                for (; i <= length - 4; i += 4)
                {
                    var diff = Vector128.Subtract(Vector128.Load(pA + i), Vector128.Load(pB + i));
                    acc = Vector128.MultiplyAddEstimate(diff, diff, acc);
                }
                sum += Vector128.Sum(acc);
            }

            for (; i < length; i++)
            {
                float diff = pA[i] - pB[i];
                sum += diff * diff;
            }
        }

        return sum;
    }
}
