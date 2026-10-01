using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Glacier.ML.Core;

/// <summary>
/// High-performance contiguous 2D feature matrix designed for zero-copy machine learning training.
/// Stored in row-major contiguous memory or zero-copy columnar memory slices to saturate CPU cache lines.
/// </summary>
public sealed unsafe class FeatureMatrix : IDisposable
{
    private float[]? _data;
    private readonly unsafe float* _nativeData;
    private readonly bool _ownsNative;
    private readonly IDisposable? _lifetimeOwner;
    private readonly ReadOnlyMemory<float>[]? _columnarMemories;
    private readonly int _rows;
    private readonly int _cols;
    private readonly string[]? _featureNames;
    private bool _disposed;

    [ThreadStatic]
    private static float[]? t_rowBuffer;

    public bool IsDisposed => _disposed;
    public int Rows => _rows;
    public int Columns => _cols;
    public string[]? FeatureNames => _featureNames;
    public bool IsColumnar => _columnarMemories != null;
    public bool IsNative => _nativeData != null;

    public unsafe float* NativePointer => _nativeData;

    public ReadOnlyMemory<float> Memory => _data != null 
        ? _data.AsMemory(0, _rows * _cols) 
        : RawArray.AsMemory(0, _rows * _cols);

    public unsafe ReadOnlySpan<float> Span
    {
        get
        {
            if (_nativeData != null)
                return new ReadOnlySpan<float>(_nativeData, _rows * _cols);
            if (_data != null)
                return _data.AsSpan(0, _rows * _cols);
            return RawArray.AsSpan(0, _rows * _cols);
        }
    }

    /// <summary>
    /// Initializes a managed row-major pinned FeatureMatrix of given dimensions.
    /// </summary>
    public FeatureMatrix(int rows, int cols, string[]? featureNames = null)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows), "Rows must be positive.");
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols), "Columns must be positive.");

        _rows = rows;
        _cols = cols;
        _data = GC.AllocateArray<float>(rows * cols, pinned: true);
        _featureNames = featureNames;
    }

    /// <summary>
    /// Wraps an existing managed row-major float array without copying.
    /// </summary>
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

    /// <summary>
    /// Zero-copy initialization from an array of columnar memory slices (e.g. from Polaris Series).
    /// </summary>
    public FeatureMatrix(ReadOnlyMemory<float>[] columns, string[]? featureNames = null)
    {
        if (columns == null || columns.Length == 0)
            throw new ArgumentException("At least one column must be provided.", nameof(columns));

        _cols = columns.Length;
        _rows = columns[0].Length;
        for (int c = 1; c < _cols; c++)
        {
            if (columns[c].Length != _rows)
                throw new ArgumentException($"Column at index {c} has length {columns[c].Length}, expected {_rows}.");
        }

        _columnarMemories = columns;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Zero-copy initialization from an array of writable columnar memory slices.
    /// </summary>
    public FeatureMatrix(Memory<float>[] columns, string[]? featureNames = null)
    {
        if (columns == null || columns.Length == 0)
            throw new ArgumentException("At least one column must be provided.", nameof(columns));

        _cols = columns.Length;
        _rows = columns[0].Length;
        var roColumns = new ReadOnlyMemory<float>[_cols];
        for (int c = 0; c < _cols; c++)
        {
            if (columns[c].Length != _rows)
                throw new ArgumentException($"Column at index {c} has length {columns[c].Length}, expected {_rows}.");
            roColumns[c] = columns[c];
        }

        _columnarMemories = roColumns;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Wraps unmanaged row-major memory pointer directly without GC allocations.
    /// </summary>
    public unsafe FeatureMatrix(
        float* nativeData, 
        int rows, 
        int cols, 
        bool ownsMemory = false, 
        IDisposable? lifetimeOwner = null, 
        string[]? featureNames = null)
    {
        if (nativeData == null) throw new ArgumentNullException(nameof(nativeData));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));

        _nativeData = nativeData;
        _rows = rows;
        _cols = cols;
        _ownsNative = ownsMemory;
        _lifetimeOwner = lifetimeOwner;
        _featureNames = featureNames;
    }

    /// <summary>
    /// Returns the contiguous column span in O(1) for columnar FeatureMatrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> GetColumn(int colIndex)
    {
        if ((uint)colIndex >= (uint)_cols)
            throw new ArgumentOutOfRangeException(nameof(colIndex));

        if (_columnarMemories != null)
        {
            return _columnarMemories[colIndex].Span;
        }

        throw new InvalidOperationException("FeatureMatrix is stored in row-major layout; use CopyColumn to extract non-contiguous column data.");
    }

    /// <summary>
    /// Returns the column memory slice for columnar FeatureMatrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlyMemory<float> GetColumnMemory(int colIndex)
    {
        if ((uint)colIndex >= (uint)_cols)
            throw new ArgumentOutOfRangeException(nameof(colIndex));

        if (_columnarMemories != null)
        {
            return _columnarMemories[colIndex];
        }

        throw new InvalidOperationException("FeatureMatrix is not stored in columnar layout.");
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe ReadOnlySpan<float> GetRow(int rowIndex)
    {
        if ((uint)rowIndex >= (uint)_rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));

        if (_columnarMemories != null)
        {
            if (t_rowBuffer == null || t_rowBuffer.Length < _cols)
            {
                t_rowBuffer = new float[_cols];
            }
            for (int c = 0; c < _cols; c++)
            {
                t_rowBuffer[c] = _columnarMemories[c].Span[rowIndex];
            }
            return new ReadOnlySpan<float>(t_rowBuffer, 0, _cols);
        }
        else if (_nativeData != null)
        {
            return new ReadOnlySpan<float>(_nativeData + (long)rowIndex * _cols, _cols);
        }
        else
        {
            return _data.AsSpan(rowIndex * _cols, _cols);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe Span<float> GetRowSpan(int rowIndex)
    {
        if ((uint)rowIndex >= (uint)_rows)
            throw new ArgumentOutOfRangeException(nameof(rowIndex));

        if (_columnarMemories != null)
        {
            if (t_rowBuffer == null || t_rowBuffer.Length < _cols)
            {
                t_rowBuffer = new float[_cols];
            }
            for (int c = 0; c < _cols; c++)
            {
                t_rowBuffer[c] = _columnarMemories[c].Span[rowIndex];
            }
            return new Span<float>(t_rowBuffer, 0, _cols);
        }
        else if (_nativeData != null)
        {
            return new Span<float>(_nativeData + (long)rowIndex * _cols, _cols);
        }
        else
        {
            return _data.AsSpan(rowIndex * _cols, _cols);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe float GetValue(int row, int col)
    {
        if (_columnarMemories != null)
        {
            return _columnarMemories[col].Span[row];
        }
        else if (_nativeData != null)
        {
            return _nativeData[(long)row * _cols + col];
        }
        else
        {
            return _data![row * _cols + col];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe void SetValue(int row, int col, float value)
    {
        if (_columnarMemories != null)
        {
            throw new InvalidOperationException("FeatureMatrix is backed by read-only columnar memory.");
        }
        else if (_nativeData != null)
        {
            _nativeData[(long)row * _cols + col] = value;
        }
        else
        {
            _data![row * _cols + col] = value;
        }
    }

    /// <summary>
    /// Extracts a single column into a destination buffer with zero heap allocation.
    /// </summary>
    public unsafe void CopyColumn(int colIndex, Span<float> destination)
    {
        if ((uint)colIndex >= (uint)_cols)
            throw new ArgumentOutOfRangeException(nameof(colIndex));
        if (destination.Length < _rows)
            throw new ArgumentException("Destination span is too small.", nameof(destination));

        if (_columnarMemories != null)
        {
            _columnarMemories[colIndex].Span.CopyTo(destination);
        }
        else if (_nativeData != null)
        {
            for (int r = 0; r < _rows; r++)
            {
                destination[r] = _nativeData[(long)r * _cols + colIndex];
            }
        }
        else
        {
            for (int r = 0; r < _rows; r++)
            {
                destination[r] = _data![r * _cols + colIndex];
            }
        }
    }

    /// <summary>
    /// Returns the underlying managed array, or materializes one on-demand if backed by columnar/native memory.
    /// </summary>
    public unsafe float[] RawArray
    {
        get
        {
            if (_data != null) return _data;

            var arr = GC.AllocateArray<float>(_rows * _cols, pinned: true);
            if (_columnarMemories != null)
            {
                for (int c = 0; c < _cols; c++)
                {
                    var colSpan = _columnarMemories[c].Span;
                    for (int r = 0; r < _rows; r++)
                    {
                        arr[r * _cols + c] = colSpan[r];
                    }
                }
            }
            else if (_nativeData != null)
            {
                fixed (float* pArr = arr)
                {
                    Buffer.MemoryCopy(_nativeData, pArr, (long)_rows * _cols * sizeof(float), (long)_rows * _cols * sizeof(float));
                }
            }

            _data = arr;
            return arr;
        }
    }

    public unsafe void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_ownsNative && _nativeData != null)
            {
                NativeMemory.AlignedFree(_nativeData);
            }
            _lifetimeOwner?.Dispose();
        }
    }
}
