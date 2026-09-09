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
        if (outputMatrix.Columns < startCol + _numCategories)
            throw new ArgumentException("Output matrix does not have enough columns.");

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
