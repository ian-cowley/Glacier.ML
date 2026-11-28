using System;
using System.Collections.Generic;
using Glacier.ML.Core;

namespace Glacier.ML.Preprocessing;

/// <summary>
/// Encodes categorical integer or string feature values into one-hot binary indicator vectors.
/// </summary>
public sealed class OneHotEncoder
{
    private readonly Dictionary<int, int> _categoryToIndex = new();
    private readonly Dictionary<string, int> _stringCategoryToIndex = new(StringComparer.Ordinal);
    private int _nullStringIndex = -1;
    private int _numCategories;
    private bool _isStringMode;
    private bool _isFitted;

    public int NumCategories => _numCategories;

    public void Fit(ReadOnlySpan<int> categories)
    {
        _categoryToIndex.Clear();
        _stringCategoryToIndex.Clear();
        _nullStringIndex = -1;
        _isStringMode = false;

        int idx = 0;
        for (int i = 0; i < categories.Length; i++)
        {
            int cat = categories[i];
            if (!_categoryToIndex.ContainsKey(cat))
            {
                _categoryToIndex[cat] = idx++;
            }
        }
        _numCategories = idx;
        _isFitted = true;
    }

    public void Fit(ReadOnlySpan<string?> categories)
    {
        _categoryToIndex.Clear();
        _stringCategoryToIndex.Clear();
        _nullStringIndex = -1;
        _isStringMode = true;

        int idx = 0;
        for (int i = 0; i < categories.Length; i++)
        {
            string? cat = categories[i];
            if (cat == null)
            {
                if (_nullStringIndex < 0)
                {
                    _nullStringIndex = idx++;
                }
            }
            else if (!_stringCategoryToIndex.ContainsKey(cat))
            {
                _stringCategoryToIndex[cat] = idx++;
            }
        }
        _numCategories = idx;
        _isFitted = true;
    }

    public void Transform(ReadOnlySpan<int> categories, FeatureMatrix outputMatrix, int startCol)
    {
        ArgumentNullException.ThrowIfNull(outputMatrix);
        if (!_isFitted)
            throw new InvalidOperationException("OneHotEncoder must be fit before calling Transform.");
        if (_isStringMode)
            throw new InvalidOperationException("OneHotEncoder was fit on string categories; call Transform(ReadOnlySpan<string?>, ...).");
        if (startCol < 0 || outputMatrix.Columns < startCol + _numCategories)
            throw new ArgumentException("Output matrix does not have enough columns.", nameof(outputMatrix));
        if (outputMatrix.Rows < categories.Length)
            throw new ArgumentException("Output matrix does not have enough rows for categories.", nameof(outputMatrix));

        if (outputMatrix.IsColumnar)
        {
            if (!outputMatrix.IsWritableColumnar)
                throw new InvalidOperationException("Cannot transform into a read-only columnar FeatureMatrix. Construct FeatureMatrix with writable Memory<float>[] columns or a row-major buffer.");

            Memory<float>[] colMemories = new Memory<float>[_numCategories];
            for (int c = 0; c < _numCategories; c++)
            {
                colMemories[c] = outputMatrix.GetWritableColumnMemory(startCol + c);
                colMemories[c].Span.Slice(0, categories.Length).Clear();
            }

            for (int r = 0; r < categories.Length; r++)
            {
                if (_categoryToIndex.TryGetValue(categories[r], out int colOffset))
                {
                    colMemories[colOffset].Span[r] = 1.0f;
                }
            }
            return;
        }

        for (int r = 0; r < categories.Length; r++)
        {
            Span<float> row = outputMatrix.GetRowSpan(r);
            row.Slice(startCol, _numCategories).Clear();

            if (_categoryToIndex.TryGetValue(categories[r], out int colOffset))
            {
                row[startCol + colOffset] = 1.0f;
            }
        }
    }

    public void Transform(ReadOnlySpan<string?> categories, FeatureMatrix outputMatrix, int startCol)
    {
        ArgumentNullException.ThrowIfNull(outputMatrix);
        if (!_isFitted)
            throw new InvalidOperationException("OneHotEncoder must be fit before calling Transform.");
        if (!_isStringMode)
            throw new InvalidOperationException("OneHotEncoder was fit on integer categories; call Transform(ReadOnlySpan<int>, ...).");
        if (startCol < 0 || outputMatrix.Columns < startCol + _numCategories)
            throw new ArgumentException("Output matrix does not have enough columns.", nameof(outputMatrix));
        if (outputMatrix.Rows < categories.Length)
            throw new ArgumentException("Output matrix does not have enough rows for categories.", nameof(outputMatrix));

        if (outputMatrix.IsColumnar)
        {
            if (!outputMatrix.IsWritableColumnar)
                throw new InvalidOperationException("Cannot transform into a read-only columnar FeatureMatrix. Construct FeatureMatrix with writable Memory<float>[] columns or a row-major buffer.");

            Memory<float>[] colMemories = new Memory<float>[_numCategories];
            for (int c = 0; c < _numCategories; c++)
            {
                colMemories[c] = outputMatrix.GetWritableColumnMemory(startCol + c);
                colMemories[c].Span.Slice(0, categories.Length).Clear();
            }

            for (int r = 0; r < categories.Length; r++)
            {
                string? cat = categories[r];
                int colOffset = cat == null
                    ? _nullStringIndex
                    : (_stringCategoryToIndex.TryGetValue(cat, out int idx) ? idx : -1);
                if (colOffset >= 0)
                {
                    colMemories[colOffset].Span[r] = 1.0f;
                }
            }
            return;
        }

        for (int r = 0; r < categories.Length; r++)
        {
            Span<float> row = outputMatrix.GetRowSpan(r);
            row.Slice(startCol, _numCategories).Clear();

            string? cat = categories[r];
            int colOffset = cat == null
                ? _nullStringIndex
                : (_stringCategoryToIndex.TryGetValue(cat, out int idx) ? idx : -1);
            if (colOffset >= 0)
            {
                row[startCol + colOffset] = 1.0f;
            }
        }
    }
}
