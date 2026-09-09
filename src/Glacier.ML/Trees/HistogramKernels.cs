using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Glacier.ML.Trees;

/// <summary>
/// High-performance SIMD-accelerated histogram building kernels.
/// Breaks CPU write-port read-after-write (RAW) dependency stalls using 4-way unrolled stack sub-histograms.
/// </summary>
public static unsafe class HistogramKernels
{
    /// <summary>
    /// Discretizes continuous float values into 256 discrete byte bins based on pre-calculated bin thresholds.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DiscretizeFeature(
        ReadOnlySpan<float> values, 
        ReadOnlySpan<float> binThresholds, 
        Span<byte> outputBins)
    {
        int length = values.Length;
        int numThresholds = binThresholds.Length;

        fixed (float* pVal = values)
        fixed (float* pThresh = binThresholds)
        fixed (byte* pOut = outputBins)
        {
            for (int i = 0; i < length; i++)
            {
                float v = pVal[i];
                // Linear / binary search for bin index
                byte bin = 0;
                while (bin < numThresholds && v > pThresh[bin])
                {
                    bin++;
                }
                pOut[i] = bin;
            }
        }
    }

    /// <summary>
    /// Builds gradient/weight histograms over binned features with 4-way unrolling and SIMD vector reduction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void BuildHistogram(
        ReadOnlySpan<byte> binnedFeature, 
        ReadOnlySpan<float> gradients, 
        Span<float> outputHistogramGradients)
    {
        outputHistogramGradients.Clear();
        int length = binnedFeature.Length;

        // 4 sub-histograms allocated on the stack (Resides entirely in L1d cache)
        Span<float> localHist = stackalloc float[256 * 4];
        localHist.Clear();

        fixed (byte* pBins = binnedFeature)
        fixed (float* pGrad = gradients)
        fixed (float* pSub = localHist)
        fixed (float* pHist = outputHistogramGradients)
        {
            int i = 0;
            // 4-way unrolled loop to saturate superscalar execution units
            for (; i <= length - 4; i += 4)
            {
                pSub[(0 * 256) + pBins[i + 0]] += pGrad[i + 0];
                pSub[(1 * 256) + pBins[i + 1]] += pGrad[i + 1];
                pSub[(2 * 256) + pBins[i + 2]] += pGrad[i + 2];
                pSub[(3 * 256) + pBins[i + 3]] += pGrad[i + 3];
            }

            // Remainder tail loop
            for (; i < length; i++)
            {
                pSub[pBins[i]] += pGrad[i];
            }

            // SIMD Vectorized reduction across the 4 sub-histograms into output buffer
            if (Avx512F.IsSupported)
            {
                for (int b = 0; b < 256; b += 16)
                {
                    var h0 = Vector512.Load(pSub + (0 * 256) + b);
                    var h1 = Vector512.Load(pSub + (1 * 256) + b);
                    var h2 = Vector512.Load(pSub + (2 * 256) + b);
                    var h3 = Vector512.Load(pSub + (3 * 256) + b);

                    var sum = Vector512.Add(Vector512.Add(h0, h1), Vector512.Add(h2, h3));
                    Vector512.Store(sum, pHist + b);
                }
            }
            else if (Avx2.IsSupported)
            {
                for (int b = 0; b < 256; b += 8)
                {
                    var h0 = Vector256.Load(pSub + (0 * 256) + b);
                    var h1 = Vector256.Load(pSub + (1 * 256) + b);
                    var h2 = Vector256.Load(pSub + (2 * 256) + b);
                    var h3 = Vector256.Load(pSub + (3 * 256) + b);

                    var sum = Vector256.Add(Vector256.Add(h0, h1), Vector256.Add(h2, h3));
                    Vector256.Store(sum, pHist + b);
                }
            }
            else if (AdvSimd.IsSupported)
            {
                for (int b = 0; b < 256; b += 4)
                {
                    var h0 = Vector128.Load(pSub + (0 * 256) + b);
                    var h1 = Vector128.Load(pSub + (1 * 256) + b);
                    var h2 = Vector128.Load(pSub + (2 * 256) + b);
                    var h3 = Vector128.Load(pSub + (3 * 256) + b);

                    var sum = Vector128.Add(Vector128.Add(h0, h1), Vector128.Add(h2, h3));
                    Vector128.Store(sum, pHist + b);
                }
            }
            else
            {
                for (int b = 0; b < 256; b++)
                {
                    pHist[b] = pSub[b] + pSub[256 + b] + pSub[512 + b] + pSub[768 + b];
                }
            }
        }
    }

    /// <summary>
    /// Builds gradient/weight and count histograms simultaneously over binned features with SIMD reduction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void BuildHistogram(
        ReadOnlySpan<byte> binnedFeature, 
        ReadOnlySpan<float> gradients, 
        Span<float> outputHistogramGradients,
        Span<int> outputHistogramCounts)
    {
        outputHistogramGradients.Clear();
        outputHistogramCounts.Clear();
        int length = binnedFeature.Length;

        Span<float> localHistG = stackalloc float[256 * 4];
        Span<int> localHistC = stackalloc int[256 * 4];
        localHistG.Clear();
        localHistC.Clear();

        fixed (byte* pBins = binnedFeature)
        fixed (float* pGrad = gradients)
        fixed (float* pSubG = localHistG)
        fixed (int* pSubC = localHistC)
        fixed (float* pHistG = outputHistogramGradients)
        fixed (int* pHistC = outputHistogramCounts)
        {
            int i = 0;
            for (; i <= length - 4; i += 4)
            {
                pSubG[(0 * 256) + pBins[i + 0]] += pGrad[i + 0];
                pSubC[(0 * 256) + pBins[i + 0]]++;

                pSubG[(1 * 256) + pBins[i + 1]] += pGrad[i + 1];
                pSubC[(1 * 256) + pBins[i + 1]]++;

                pSubG[(2 * 256) + pBins[i + 2]] += pGrad[i + 2];
                pSubC[(2 * 256) + pBins[i + 2]]++;

                pSubG[(3 * 256) + pBins[i + 3]] += pGrad[i + 3];
                pSubC[(3 * 256) + pBins[i + 3]]++;
            }

            for (; i < length; i++)
            {
                pSubG[pBins[i]] += pGrad[i];
                pSubC[pBins[i]]++;
            }

            // Reduction across the 4 sub-histograms
            if (Avx2.IsSupported)
            {
                for (int b = 0; b < 256; b += 8)
                {
                    var g0 = Vector256.Load(pSubG + (0 * 256) + b);
                    var g1 = Vector256.Load(pSubG + (1 * 256) + b);
                    var g2 = Vector256.Load(pSubG + (2 * 256) + b);
                    var g3 = Vector256.Load(pSubG + (3 * 256) + b);
                    var sumG = Vector256.Add(Vector256.Add(g0, g1), Vector256.Add(g2, g3));
                    Vector256.Store(sumG, pHistG + b);

                    var c0 = Vector256.Load(pSubC + (0 * 256) + b);
                    var c1 = Vector256.Load(pSubC + (1 * 256) + b);
                    var c2 = Vector256.Load(pSubC + (2 * 256) + b);
                    var c3 = Vector256.Load(pSubC + (3 * 256) + b);
                    var sumC = Vector256.Add(Vector256.Add(c0, c1), Vector256.Add(c2, c3));
                    Vector256.Store(sumC, pHistC + b);
                }
            }
            else
            {
                for (int b = 0; b < 256; b++)
                {
                    pHistG[b] = pSubG[b] + pSubG[256 + b] + pSubG[512 + b] + pSubG[768 + b];
                    pHistC[b] = pSubC[b] + pSubC[256 + b] + pSubC[512 + b] + pSubC[768 + b];
                }
            }
        }
    }
}
