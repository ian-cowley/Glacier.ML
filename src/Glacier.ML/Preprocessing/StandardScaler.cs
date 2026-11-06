using System;
using System.Runtime.CompilerServices;
using Glacier.ML.Core;

namespace Glacier.ML.Preprocessing;

/// <summary>
/// Standardizes features by removing the mean and scaling to unit variance: z = (x - u) / s.
/// Operates directly on contiguous memory with zero allocations during transform.
/// </summary>
public sealed class StandardScaler : ITransformer
{
    private float[]? _means;
    private float[]? _stdDevs;

    public float[]? Means => _means;
    public float[]? StdDevs => _stdDevs;

    public void Fit(FeatureMatrix matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        _means = new float[cols];
        _stdDevs = new float[cols];

        for (int c = 0; c < cols; c++)
        {
            double sum = 0;
            if (matrix.IsColumnar)
            {
                ReadOnlySpan<float> col = matrix.GetColumn(c);
                for (int r = 0; r < rows; r++)
                {
                    sum += col[r];
                }
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    sum += matrix.GetValue(r, c);
                }
            }

            float mean = (float)(sum / rows);
            _means[c] = mean;

            double sumSqDiff = 0;
            if (matrix.IsColumnar)
            {
                ReadOnlySpan<float> col = matrix.GetColumn(c);
                for (int r = 0; r < rows; r++)
                {
                    float diff = col[r] - mean;
                    sumSqDiff += diff * diff;
                }
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    float diff = matrix.GetValue(r, c) - mean;
                    sumSqDiff += diff * diff;
                }
            }

            float std = (float)Math.Sqrt(sumSqDiff / rows);
            _stdDevs[c] = std > 1e-8f ? std : 1f; // Avoid division by zero
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Transform(FeatureMatrix matrix)
    {
        if (_means == null || _stdDevs == null)
            throw new InvalidOperationException("StandardScaler must be fit before transforming data.");

        int rows = matrix.Rows;
        int cols = Math.Min(matrix.Columns, _means.Length);

        if (matrix.IsColumnar)
        {
            if (!matrix.IsWritableColumnar)
                throw new InvalidOperationException("Cannot transform a read-only columnar FeatureMatrix in-place. Construct FeatureMatrix with writable Memory<float>[] columns or a row-major buffer.");

            for (int c = 0; c < cols; c++)
            {
                Span<float> col = matrix.GetWritableColumn(c);
                float mean = _means[c];
                float invStd = 1.0f / _stdDevs[c];
                for (int r = 0; r < rows; r++)
                {
                    col[r] = (col[r] - mean) * invStd;
                }
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
        if (_means == null || _stdDevs == null)
            throw new InvalidOperationException("StandardScaler must be fit before transforming data.");

        int len = Math.Min(input.Length, _means.Length);
        for (int c = 0; c < len; c++)
        {
            output[c] = (input[c] - _means[c]) / _stdDevs[c];
        }
    }
}
