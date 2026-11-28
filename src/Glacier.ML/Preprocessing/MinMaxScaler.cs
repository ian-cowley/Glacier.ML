using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Glacier.ML.Core;

namespace Glacier.ML.Preprocessing;

/// <summary>
/// Transforms features by scaling each feature to a given range, typically [0, 1],
/// using SIMD vectorization across contiguous or writable columnar memory.
/// </summary>
public sealed class MinMaxScaler : ITransformer, IUnsupervisedEstimator<MinMaxScaler>
{
    private float[]? _mins;
    private float[]? _maxs;
    private float[]? _ranges;
    private float[]? _invRanges;

    public float[]? Mins => _mins;
    public float[]? Maxs => _maxs;

    MinMaxScaler IUnsupervisedEstimator<MinMaxScaler>.Fit(FeatureMatrix features)
    {
        Fit(features);
        return this;
    }

    public void Fit(FeatureMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        if (rows <= 0)
            throw new ArgumentException("FeatureMatrix must contain at least one row.", nameof(matrix));

        _mins = new float[cols];
        _maxs = new float[cols];
        _ranges = new float[cols];
        _invRanges = new float[cols];

        if (matrix.IsColumnar)
        {
            for (int c = 0; c < cols; c++)
            {
                ComputeColumnMinMaxSimd(matrix.GetColumn(c), out float minVal, out float maxVal);
                _mins[c] = minVal;
                _maxs[c] = maxVal;
                float range = maxVal - minVal;
                float safeRange = range > 1e-8f ? range : 1f;
                _ranges[c] = safeRange;
                _invRanges[c] = 1.0f / safeRange;
            }
            return;
        }

        // Sequential row-major streaming pass (avoids strided GetValue calls)
        Array.Fill(_mins, float.PositiveInfinity);
        Array.Fill(_maxs, float.NegativeInfinity);
        ReadOnlySpan<float> data = matrix.Span;

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = data.Slice((int)((long)r * cols), cols);
            AccumulateRowMinMaxSimd(row, _mins, _maxs);
        }

        for (int c = 0; c < cols; c++)
        {
            float range = _maxs[c] - _mins[c];
            float safeRange = range > 1e-8f ? range : 1f;
            _ranges[c] = safeRange;
            _invRanges[c] = 1.0f / safeRange;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Transform(FeatureMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (_mins == null || _invRanges == null)
            throw new InvalidOperationException("MinMaxScaler must be fit before transforming data.");

        int rows = matrix.Rows;
        int cols = Math.Min(matrix.Columns, _mins.Length);
        if (rows == 0) return;

        if (matrix.IsColumnar)
        {
            if (!matrix.IsWritableColumnar)
                throw new InvalidOperationException("Cannot transform a read-only columnar FeatureMatrix in-place. Construct FeatureMatrix with writable Memory<float>[] columns or a row-major buffer.");

            for (int c = 0; c < cols; c++)
            {
                Span<float> col = matrix.GetWritableColumn(c);
                TransformColumnSimd(col.Slice(0, rows), _mins[c], _invRanges[c]);
            }
            return;
        }

        for (int r = 0; r < rows; r++)
        {
            Span<float> row = matrix.GetRowSpan(r);
            TransformRow(row, row);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void TransformRow(ReadOnlySpan<float> input, Span<float> output)
    {
        if (_mins == null || _invRanges == null)
            throw new InvalidOperationException("MinMaxScaler must be fit before transforming data.");

        int len = Math.Min(input.Length, _mins.Length);
        if (output.Length < len)
            throw new ArgumentException("Output span is smaller than input length.", nameof(output));

        int i = 0;
        ref float rIn = ref MemoryMarshal.GetReference(input);
        ref float rOut = ref MemoryMarshal.GetReference(output);
        ref float rMin = ref MemoryMarshal.GetArrayDataReference(_mins);
        ref float rInv = ref MemoryMarshal.GetArrayDataReference(_invRanges);

        if (Vector512.IsHardwareAccelerated && len >= Vector512<float>.Count)
        {
            int step = Vector512<float>.Count;
            int limit = len - step;
            while (i <= limit)
            {
                var vx = Vector512.LoadUnsafe(ref rIn, (nuint)i);
                var vmin = Vector512.LoadUnsafe(ref rMin, (nuint)i);
                var vinv = Vector512.LoadUnsafe(ref rInv, (nuint)i);
                ((vx - vmin) * vinv).StoreUnsafe(ref rOut, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && len >= Vector256<float>.Count)
        {
            int step = Vector256<float>.Count;
            int limit = len - step;
            while (i <= limit)
            {
                var vx = Vector256.LoadUnsafe(ref rIn, (nuint)i);
                var vmin = Vector256.LoadUnsafe(ref rMin, (nuint)i);
                var vinv = Vector256.LoadUnsafe(ref rInv, (nuint)i);
                ((vx - vmin) * vinv).StoreUnsafe(ref rOut, (nuint)i);
                i += step;
            }
        }
        else if (Vector128.IsHardwareAccelerated && len >= Vector128<float>.Count)
        {
            int step = Vector128<float>.Count;
            int limit = len - step;
            while (i <= limit)
            {
                var vx = Vector128.LoadUnsafe(ref rIn, (nuint)i);
                var vmin = Vector128.LoadUnsafe(ref rMin, (nuint)i);
                var vinv = Vector128.LoadUnsafe(ref rInv, (nuint)i);
                ((vx - vmin) * vinv).StoreUnsafe(ref rOut, (nuint)i);
                i += step;
            }
        }

        for (; i < len; i++)
        {
            output[i] = (input[i] - _mins[i]) * _invRanges[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TransformColumnSimd(Span<float> col, float minVal, float invRange)
    {
        int length = col.Length;
        int i = 0;
        ref float rCol = ref MemoryMarshal.GetReference(col);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var vmin = Vector512.Create(minVal);
            var vinv = Vector512.Create(invRange);
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vx = Vector512.LoadUnsafe(ref rCol, (nuint)i);
                ((vx - vmin) * vinv).StoreUnsafe(ref rCol, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var vmin = Vector256.Create(minVal);
            var vinv = Vector256.Create(invRange);
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vx = Vector256.LoadUnsafe(ref rCol, (nuint)i);
                ((vx - vmin) * vinv).StoreUnsafe(ref rCol, (nuint)i);
                i += step;
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            var vmin = Vector128.Create(minVal);
            var vinv = Vector128.Create(invRange);
            int step = Vector128<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vx = Vector128.LoadUnsafe(ref rCol, (nuint)i);
                ((vx - vmin) * vinv).StoreUnsafe(ref rCol, (nuint)i);
                i += step;
            }
        }

        for (; i < length; i++)
        {
            col[i] = (col[i] - minVal) * invRange;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeColumnMinMaxSimd(ReadOnlySpan<float> col, out float minVal, out float maxVal)
    {
        int length = col.Length;
        minVal = float.PositiveInfinity;
        maxVal = float.NegativeInfinity;
        int i = 0;
        ref float rCol = ref MemoryMarshal.GetReference(col);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var vmin = Vector512.Create(float.PositiveInfinity);
            var vmax = Vector512.Create(float.NegativeInfinity);
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var v = Vector512.LoadUnsafe(ref rCol, (nuint)i);
                vmin = Vector512.Min(vmin, v);
                vmax = Vector512.Max(vmax, v);
                i += step;
            }
            for (int lane = 0; lane < step; lane++)
            {
                if (vmin[lane] < minVal) minVal = vmin[lane];
                if (vmax[lane] > maxVal) maxVal = vmax[lane];
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var vmin = Vector256.Create(float.PositiveInfinity);
            var vmax = Vector256.Create(float.NegativeInfinity);
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var v = Vector256.LoadUnsafe(ref rCol, (nuint)i);
                vmin = Vector256.Min(vmin, v);
                vmax = Vector256.Max(vmax, v);
                i += step;
            }
            for (int lane = 0; lane < step; lane++)
            {
                if (vmin[lane] < minVal) minVal = vmin[lane];
                if (vmax[lane] > maxVal) maxVal = vmax[lane];
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            var vmin = Vector128.Create(float.PositiveInfinity);
            var vmax = Vector128.Create(float.NegativeInfinity);
            int step = Vector128<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var v = Vector128.LoadUnsafe(ref rCol, (nuint)i);
                vmin = Vector128.Min(vmin, v);
                vmax = Vector128.Max(vmax, v);
                i += step;
            }
            for (int lane = 0; lane < step; lane++)
            {
                if (vmin[lane] < minVal) minVal = vmin[lane];
                if (vmax[lane] > maxVal) maxVal = vmax[lane];
            }
        }

        for (; i < length; i++)
        {
            float v = col[i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateRowMinMaxSimd(ReadOnlySpan<float> row, Span<float> mins, Span<float> maxs)
    {
        int len = row.Length;
        int i = 0;
        ref float rRow = ref MemoryMarshal.GetReference(row);
        ref float rMin = ref MemoryMarshal.GetReference(mins);
        ref float rMax = ref MemoryMarshal.GetReference(maxs);

        if (Vector512.IsHardwareAccelerated && len >= Vector512<float>.Count)
        {
            int step = Vector512<float>.Count;
            int limit = len - step;
            while (i <= limit)
            {
                var v = Vector512.LoadUnsafe(ref rRow, (nuint)i);
                Vector512.Min(Vector512.LoadUnsafe(ref rMin, (nuint)i), v).StoreUnsafe(ref rMin, (nuint)i);
                Vector512.Max(Vector512.LoadUnsafe(ref rMax, (nuint)i), v).StoreUnsafe(ref rMax, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && len >= Vector256<float>.Count)
        {
            int step = Vector256<float>.Count;
            int limit = len - step;
            while (i <= limit)
            {
                var v = Vector256.LoadUnsafe(ref rRow, (nuint)i);
                Vector256.Min(Vector256.LoadUnsafe(ref rMin, (nuint)i), v).StoreUnsafe(ref rMin, (nuint)i);
                Vector256.Max(Vector256.LoadUnsafe(ref rMax, (nuint)i), v).StoreUnsafe(ref rMax, (nuint)i);
                i += step;
            }
        }
        else if (Vector128.IsHardwareAccelerated && len >= Vector128<float>.Count)
        {
            int step = Vector128<float>.Count;
            int limit = len - step;
            while (i <= limit)
            {
                var v = Vector128.LoadUnsafe(ref rRow, (nuint)i);
                Vector128.Min(Vector128.LoadUnsafe(ref rMin, (nuint)i), v).StoreUnsafe(ref rMin, (nuint)i);
                Vector128.Max(Vector128.LoadUnsafe(ref rMax, (nuint)i), v).StoreUnsafe(ref rMax, (nuint)i);
                i += step;
            }
        }

        for (; i < len; i++)
        {
            float v = row[i];
            if (v < mins[i]) mins[i] = v;
            if (v > maxs[i]) maxs[i] = v;
        }
    }
}
