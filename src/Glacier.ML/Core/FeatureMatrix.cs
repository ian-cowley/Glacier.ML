using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Glacier.ML.Core;

/// <summary>
/// High-performance contiguous 2D feature matrix designed for zero-copy machine learning training.
/// Stored in row-major or column-major contiguous memory to saturate CPU cache lines.
/// </summary>
public sealed class FeatureMatrix : IDisposable
{
    private readonly float[] _data;
    private readonly int _rows;
    private readonly int _cols;
    private readonly string[]? _featureNames;
    private bool _disposed;

    public bool IsDisposed => _disposed;
    public int Rows => _rows;
    public int Columns => _cols;
    public string[]? FeatureNames => _featureNames;
    public ReadOnlyMemory<float> Memory => _data.AsMemory(0, _rows * _cols);
    public ReadOnlySpan<float> Span => _data.AsSpan(0, _rows * _cols);

    public FeatureMatrix(int rows, int cols, string[]? featureNames = null)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows), "Rows must be positive.");
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols), "Columns must be positive.");

        _rows = rows;
        _cols = cols;
        _data = GC.AllocateArray<float>(rows * cols, pinned: true);
        _featureNames = featureNames;
    }

    public FeatureMatrix(float[] data, int rows, int cols, string[]? featureNames = null)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        if (rows <= 0 || cols <= 0 || data.Length < rows * cols)
            throw new ArgumentException("Data length does not match specified dimensions.");

        _rows = rows;
        _cols = cols;
        _data = data;
        _featureNames = featureNames;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> GetRow(int rowIndex)
    {
        if ((uint)rowIndex >= (uint)_rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));

        return _data.AsSpan(rowIndex * _cols, _cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRowSpan(int rowIndex)
    {
        if ((uint)rowIndex >= (uint)_rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));

        return _data.AsSpan(rowIndex * _cols, _cols);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float GetValue(int row, int col)
    {
        return _data[row * _cols + col];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetValue(int row, int col, float value)
    {
        _data[row * _cols + col] = value;
    }

    /// <summary>
    /// Extracts a single column into a destination buffer with zero heap allocation.
    /// </summary>
    public void CopyColumn(int colIndex, Span<float> destination)
    {
        if ((uint)colIndex >= (uint)_cols)
            throw new ArgumentOutOfRangeException(nameof(colIndex));
        if (destination.Length < _rows)
            throw new ArgumentException("Destination span is too small.", nameof(destination));

        for (int r = 0; r < _rows; r++)
        {
            destination[r] = _data[r * _cols + colIndex];
        }
    }

    public float[] RawArray => _data;

    public void Dispose()
    {
        _disposed = true;
    }
}
