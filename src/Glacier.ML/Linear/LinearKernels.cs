using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Glacier.ML.Linear;

public static unsafe class LinearKernels
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int length = a.Length;
        if (b.Length < length)
            throw new ArgumentException("Span b must be at least as long as span a.", nameof(b));

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
                    acc = Vector512.FusedMultiplyAdd(Vector512.Load(pA + i), Vector512.Load(pB + i), acc);
                }
                sum += Vector512.Sum(acc);
            }
            else if (Vector256.IsHardwareAccelerated && length >= 8)
            {
                var acc = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    acc = Vector256.FusedMultiplyAdd(Vector256.Load(pA + i), Vector256.Load(pB + i), acc);
                }
                sum += Vector256.Sum(acc);
            }
            else if (Vector128.IsHardwareAccelerated && length >= 4)
            {
                var acc = Vector128<float>.Zero;
                for (; i <= length - 4; i += 4)
                {
                    acc = Vector128.FusedMultiplyAdd(Vector128.Load(pA + i), Vector128.Load(pB + i), acc);
                }
                sum += Vector128.Sum(acc);
            }

            for (; i < length; i++)
            {
                sum += pA[i] * pB[i];
            }
        }

        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float z)
    {
        if (z > 20f) return 1f;
        if (z < -20f) return 0f;
        return 1.0f / (1.0f + MathF.Exp(-z));
    }
}
