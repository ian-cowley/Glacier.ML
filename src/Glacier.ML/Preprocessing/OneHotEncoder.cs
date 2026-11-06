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
    private int _numCategories;

    public int NumCategories => _numCategories;

    public void Fit(ReadOnlySpan<int> categories)
    {
        _categoryToIndex.Clear();
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
    }

    public void Transform(ReadOnlySpan<int> categories, FeatureMatrix outputMatrix, int startCol)
    {
        if (startCol < 0 || outputMatrix.Columns < startCol + _numCategories)
            throw new ArgumentException("Output matrix does not have enough columns.", nameof(outputMatrix));
        if (outputMatrix.Rows < categories.Length)
            throw new ArgumentException("Output matrix does not have enough rows for categories.", nameof(outputMatrix));

        if (outputMatrix.IsColumnar)
        {
            if (!outputMatrix.IsWritableColumnar)
                throw new InvalidOperationException("Cannot transform into a read-only columnar FeatureMatrix. Construct FeatureMatrix with writable Memory<float>[] columns or a row-major buffer.");

            for (int c = 0; c < _numCategories; c++)
            {
                outputMatrix.GetWritableColumn(startCol + c).Slice(0, categories.Length).Clear();
            }

            for (int r = 0; r < categories.Length; r++)
            {
                if (_categoryToIndex.TryGetValue(categories[r], out int colOffset))
                {
                    outputMatrix.GetWritableColumn(startCol + colOffset)[r] = 1.0f;
                }
            }
            return;
        }

        for (int r = 0; r < categories.Length; r++)
        {
            // Clear destination slice
            Span<float> row = outputMatrix.GetRowSpan(r);
            row.Slice(startCol, _numCategories).Clear();

            if (_categoryToIndex.TryGetValue(categories[r], out int colOffset))
            {
                row[startCol + colOffset] = 1.0f;
            }
        }
    }
}
