using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.ML.Core;

namespace Glacier.ML.Preprocessing;

/// <summary>
/// Standardizes features by removing the mean and scaling to unit variance: z = (x - u) / s.
/// Operates directly on contiguous or writable columnar memory with SIMD vectorization and zero allocations during transform.
/// </summary>
public sealed class StandardScaler : ITransformer, IUnsupervisedEstimator<StandardScaler>
{
    private float[]? _means;
    private float[]? _stdDevs;
    private float[]? _invStdDevs;

    public float[]? Means => _means;
    public float[]? StdDevs => _stdDevs;

    StandardScaler IUnsupervisedEstimator<StandardScaler>.Fit(FeatureMatrix features)
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

        _means = new float[cols];
        _stdDevs = new float[cols];
        _invStdDevs = new float[cols];

        if (matrix.IsColumnar)
        {
            for (int c = 0; c < cols; c++)
            {
                ReadOnlySpan<float> col = matrix.GetColumn(c);
                double sum = SumDoubleSimd(col);
                float mean = (float)(sum / rows);
                _means[c] = mean;

                double sumSqDiff = SumSquaredDiffDoubleSimd(col, mean);
                float std = (float)Math.Sqrt(sumSqDiff / rows);
                float safeStd = std > 1e-8f ? std : 1f;
                _stdDevs[c] = safeStd;
                _invStdDevs[c] = 1.0f / safeStd;
            }
            return;
        }

        // Sequential row-major streaming pass (avoids strided column accesses)
        double[] sums = new double[cols];
        double[] sumSq = new double[cols];
        ReadOnlySpan<float> data = matrix.Span;

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = data.Slice((int)((long)r * cols), cols);
            AccumulateRowDoubleSimd(row, sums);
        }

        for (int c = 0; c < cols; c++)
        {
            _means[c] = (float)(sums[c] / rows);
        }

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = data.Slice((int)((long)r * cols), cols);
            AccumulateRowSquaredDiffDoubleSimd(row, _means, sumSq);
        }

        for (int c = 0; c < cols; c++)
        {
            float std = (float)Math.Sqrt(sumSq[c] / rows);
            float safeStd = std > 1e-8f ? std : 1f;
            _stdDevs[c] = safeStd;
            _invStdDevs[c] = 1.0f / safeStd;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Transform(FeatureMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        if (_means == null || _invStdDevs == null)
            throw new InvalidOperationException("StandardScaler must be fit before transforming data.");

        int rows = matrix.Rows;
        int cols = Math.Min(matrix.Columns, _means.Length);
        if (rows == 0) return;

        if (matrix.IsColumnar)
        {
            if (!matrix.IsWritableColumnar)
                throw new InvalidOperationException("Cannot transform a read-only columnar FeatureMatrix in-place. Construct FeatureMatrix with writable Memory<float>[] columns or a row-major buffer.");

            for (int c = 0; c < cols; c++)
            {
                Span<float> col = matrix.GetWritableColumn(c);
                TransformColumnSimd(col.Slice(0, rows), _means[c], _invStdDevs[c]);
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
        if (_means == null || _invStdDevs == null)
            throw new InvalidOperationException("StandardScaler must be fit before transforming data.");

        int len = Math.Min(input.Length, _means.Length);
        if (output.Length < len)
            throw new ArgumentException("Output span is smaller than input length.", nameof(output));

        int i = 0;
        ref float rIn = ref MemoryMarshal.GetReference(input);
        ref float rOut = ref MemoryMarshal.GetReference(output);
        ref float rMean = ref MemoryMarshal.GetArrayDataReference(_means);
        ref float rInv = ref MemoryMarshal.GetArrayDataReference(_invStdDevs);

        if (Vector512.IsHardwareAccelerated && len >= Vector512<float>.Count)
        {
            int step = Vector512<float>.Count;
            int limit = len - step;
            while (i <= limit)
            {
                var vx = Vector512.LoadUnsafe(ref rIn, (nuint)i);
                var vm = Vector512.LoadUnsafe(ref rMean, (nuint)i);
                var vi = Vector512.LoadUnsafe(ref rInv, (nuint)i);
                ((vx - vm) * vi).StoreUnsafe(ref rOut, (nuint)i);
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
                var vm = Vector256.LoadUnsafe(ref rMean, (nuint)i);
                var vi = Vector256.LoadUnsafe(ref rInv, (nuint)i);
                ((vx - vm) * vi).StoreUnsafe(ref rOut, (nuint)i);
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
                var vm = Vector128.LoadUnsafe(ref rMean, (nuint)i);
                var vi = Vector128.LoadUnsafe(ref rInv, (nuint)i);
                ((vx - vm) * vi).StoreUnsafe(ref rOut, (nuint)i);
                i += step;
            }
        }

        for (; i < len; i++)
        {
            output[i] = (input[i] - _means[i]) * _invStdDevs[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void TransformColumnSimd(Span<float> col, float mean, float invStd)
    {
        int length = col.Length;
        int i = 0;
        ref float rCol = ref MemoryMarshal.GetReference(col);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var vm = Vector512.Create(mean);
            var vi = Vector512.Create(invStd);
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vx = Vector512.LoadUnsafe(ref rCol, (nuint)i);
                ((vx - vm) * vi).StoreUnsafe(ref rCol, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var vm = Vector256.Create(mean);
            var vi = Vector256.Create(invStd);
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vx = Vector256.LoadUnsafe(ref rCol, (nuint)i);
                ((vx - vm) * vi).StoreUnsafe(ref rCol, (nuint)i);
                i += step;
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            var vm = Vector128.Create(mean);
            var vi = Vector128.Create(invStd);
            int step = Vector128<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vx = Vector128.LoadUnsafe(ref rCol, (nuint)i);
                ((vx - vm) * vi).StoreUnsafe(ref rCol, (nuint)i);
                i += step;
            }
        }

        for (; i < length; i++)
        {
            col[i] = (col[i] - mean) * invStd;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double SumDoubleSimd(ReadOnlySpan<float> col)
    {
        int n = col.Length;
        int vecSize = Vector<float>.Count;
        var vAccLo = Vector<double>.Zero;
        var vAccHi = Vector<double>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            Vector.Widen(new Vector<float>(col.Slice(i, vecSize)), out var lo, out var hi);
            vAccLo += lo;
            vAccHi += hi;
        }

        double sum = Vector.Sum(vAccLo) + Vector.Sum(vAccHi);
        for (; i < n; i++) sum += col[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static double SumSquaredDiffDoubleSimd(ReadOnlySpan<float> col, double mean)
    {
        int n = col.Length;
        int vecSize = Vector<float>.Count;
        var vMean = new Vector<double>(mean);
        var vAccLo = Vector<double>.Zero;
        var vAccHi = Vector<double>.Zero;

        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            Vector.Widen(new Vector<float>(col.Slice(i, vecSize)), out var lo, out var hi);
            var dLo = lo - vMean;
            var dHi = hi - vMean;
            vAccLo += dLo * dLo;
            vAccHi += dHi * dHi;
        }

        double sum = Vector.Sum(vAccLo) + Vector.Sum(vAccHi);
        for (; i < n; i++)
        {
            double d = col[i] - mean;
            sum += d * d;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateRowDoubleSimd(ReadOnlySpan<float> row, Span<double> sums)
    {
        int n = row.Length;
        int vecSize = Vector<float>.Count;
        int dStep = Vector<double>.Count;
        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            Vector.Widen(new Vector<float>(row.Slice(i, vecSize)), out var lo, out var hi);
            (new Vector<double>(sums.Slice(i, dStep)) + lo).CopyTo(sums.Slice(i, dStep));
            (new Vector<double>(sums.Slice(i + dStep, dStep)) + hi).CopyTo(sums.Slice(i + dStep, dStep));
        }
        for (; i < n; i++) sums[i] += row[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateRowSquaredDiffDoubleSimd(ReadOnlySpan<float> row, ReadOnlySpan<float> means, Span<double> sumSq)
    {
        int n = row.Length;
        int vecSize = Vector<float>.Count;
        int dStep = Vector<double>.Count;
        int i = 0;
        for (; i <= n - vecSize; i += vecSize)
        {
            var diff = new Vector<float>(row.Slice(i, vecSize)) - new Vector<float>(means.Slice(i, vecSize));
            Vector.Widen(diff, out var lo, out var hi);
            (new Vector<double>(sumSq.Slice(i, dStep)) + lo * lo).CopyTo(sumSq.Slice(i, dStep));
            (new Vector<double>(sumSq.Slice(i + dStep, dStep)) + hi * hi).CopyTo(sumSq.Slice(i + dStep, dStep));
        }
        for (; i < n; i++)
        {
            double diff = row[i] - means[i];
            sumSq[i] += diff * diff;
        }
    }
}
