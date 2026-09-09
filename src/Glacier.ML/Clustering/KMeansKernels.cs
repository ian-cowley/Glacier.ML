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
        float* pCentroids, 
        int numCentroids, 
        int dimensions)
    {
        int bestCluster = 0;
        float minDistance = float.MaxValue;

        for (int k = 0; k < numCentroids; k++)
        {
            float* pC = pCentroids + (k * dimensions);
            float dist = 0f;
            int d = 0;

            if (Avx512F.IsSupported && dimensions >= 16)
            {
                var acc = Vector512<float>.Zero;
                for (; d <= dimensions - 16; d += 16)
                {
                    var diff = Vector512.Subtract(Vector512.Load(pSample + d), Vector512.Load(pC + d));
                    acc = Avx512F.FusedMultiplyAdd(diff, diff, acc);
                }
                dist = Vector512.Sum(acc);
            }
            else if (Avx2.IsSupported && dimensions >= 8)
            {
                var acc = Vector256<float>.Zero;
                for (; d <= dimensions - 8; d += 8)
                {
                    var diff = Vector256.Subtract(Vector256.Load(pSample + d), Vector256.Load(pC + d));
                    acc = Vector256.Add(acc, Vector256.Multiply(diff, diff));
                }
                dist = Vector256.Sum(acc);
            }
            else if (AdvSimd.IsSupported && dimensions >= 4)
            {
                var acc = Vector128<float>.Zero;
                for (; d <= dimensions - 4; d += 4)
                {
                    var diff = Vector128.Subtract(Vector128.Load(pSample + d), Vector128.Load(pC + d));
                    acc = Vector128.Add(acc, Vector128.Multiply(diff, diff));
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
            if (Vector256.IsHardwareAccelerated && length >= 8)
            {
                var acc = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    var diff = Vector256.Subtract(Vector256.Load(pA + i), Vector256.Load(pB + i));
                    acc = Vector256.Add(acc, Vector256.Multiply(diff, diff));
                }
                sum += Vector256.Sum(acc);
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
