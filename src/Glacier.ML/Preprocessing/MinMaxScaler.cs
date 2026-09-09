using System;
using System.Runtime.CompilerServices;
using Glacier.ML.Core;

namespace Glacier.ML.Preprocessing;

/// <summary>
/// Transforms features by scaling each feature to a given range, typically [0, 1].
/// </summary>
public sealed class MinMaxScaler : ITransformer
{
    private float[]? _mins;
    private float[]? _maxs;
    private float[]? _ranges;

    public float[]? Mins => _mins;
    public float[]? Maxs => _maxs;

    public void Fit(FeatureMatrix matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        _mins = new float[cols];
        _maxs = new float[cols];
        _ranges = new float[cols];

        for (int c = 0; c < cols; c++)
        {
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;

            for (int r = 0; r < rows; r++)
            {
                float v = matrix.GetValue(r, c);
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }

            _mins[c] = minVal;
            _maxs[c] = maxVal;
            float range = maxVal - minVal;
            _ranges[c] = range > 1e-8f ? range : 1f;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Transform(FeatureMatrix matrix)
    {
        if (_mins == null || _ranges == null)
            throw new InvalidOperationException("MinMaxScaler must be fit before transforming data.");

        int rows = matrix.Rows;
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = matrix.GetRowSpan(r);
            TransformRow(row, row);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void TransformRow(ReadOnlySpan<float> input, Span<float> output)
    {
        if (_mins == null || _ranges == null)
            throw new InvalidOperationException("MinMaxScaler must be fit before transforming data.");

        int len = Math.Min(input.Length, _mins.Length);
        for (int c = 0; c < len; c++)
        {
            output[c] = (input[c] - _mins[c]) / _ranges[c];
        }
    }
}
