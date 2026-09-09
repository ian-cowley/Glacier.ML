using System;
using System.Linq;
using Glacier.ML.Clustering;
using Glacier.ML.Core;
using Glacier.ML.Trees;
using Glacier.Polaris;
using Glacier.Polaris.Data;

namespace Glacier.ML.Interop;

public static class PolarisExtensions
{
    /// <summary>
    /// Converts selected numeric columns of a Polaris DataFrame into a zero-copy aligned FeatureMatrix.
    /// </summary>
    public static FeatureMatrix ToFeatureMatrix(this DataFrame df, params string[] featureColumnNames)
    {
        if (df == null) throw new ArgumentNullException(nameof(df));
        if (featureColumnNames == null || featureColumnNames.Length == 0)
            throw new ArgumentException("At least one feature column name must be provided.", nameof(featureColumnNames));

        int rows = df.RowCount;
        int cols = featureColumnNames.Length;
        var matrix = new FeatureMatrix(rows, cols, featureColumnNames);

        for (int c = 0; c < cols; c++)
        {
            string colName = featureColumnNames[c];
            var series = df.Columns.FirstOrDefault(x => x.Name.Equals(colName, StringComparison.OrdinalIgnoreCase))
                ?? throw new ArgumentException($"Column '{colName}' not found in DataFrame.");

            if (series is Series<double> doubleSeries)
            {
                var span = doubleSeries.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    matrix.SetValue(r, c, (float)span[r]);
                }
            }
            else if (series is Series<float> floatSeries)
            {
                var span = floatSeries.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    matrix.SetValue(r, c, span[r]);
                }
            }
            else if (series is Series<int> intSeries)
            {
                var span = intSeries.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    matrix.SetValue(r, c, (float)span[r]);
                }
            }
            else if (series is Series<long> longSeries)
            {
                var span = longSeries.Memory.Span;
                for (int r = 0; r < rows; r++)
                {
                    matrix.SetValue(r, c, (float)span[r]);
                }
            }
            else
            {
                throw new NotSupportedException($"Series type '{series.GetType().Name}' is not supported for FeatureMatrix conversion.");
            }
        }

        return matrix;
    }

    /// <summary>
    /// Fits a FastRandomForest model directly from a Polaris DataFrame.
    /// </summary>
    public static FastRandomForest FitRandomForest(
        this DataFrame df, 
        string targetColumn, 
        string[] featureColumns, 
        int numTrees = 100, 
        int maxDepth = 8)
    {
        var targetSeries = df.Columns.FirstOrDefault(x => x.Name.Equals(targetColumn, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException($"Target column '{targetColumn}' not found in DataFrame.");

        int rows = df.RowCount;
        float[] targets = new float[rows];

        if (targetSeries is Series<double> dt)
        {
            var span = dt.Memory.Span;
            for (int r = 0; r < rows; r++) targets[r] = (float)span[r];
        }
        else if (targetSeries is Series<int> it)
        {
            var span = it.Memory.Span;
            for (int r = 0; r < rows; r++) targets[r] = (float)span[r];
        }
        else if (targetSeries is Series<float> ft)
        {
            ft.Memory.Span.CopyTo(targets);
        }
        else
        {
            throw new NotSupportedException($"Target series type '{targetSeries.GetType().Name}' is not supported.");
        }

        var features = df.ToFeatureMatrix(featureColumns);
        var forest = new FastRandomForest(numTrees, maxDepth);
        forest.Fit(features, targets);
        return forest;
    }

    /// <summary>
    /// Fits a KMeans clustering model directly from a Polaris DataFrame.
    /// </summary>
    public static KMeans FitKMeans(
        this DataFrame df, 
        string[] featureColumns, 
        int k = 8, 
        int maxIterations = 100)
    {
        var features = df.ToFeatureMatrix(featureColumns);
        var kmeans = new KMeans(k, maxIterations);
        kmeans.Fit(features);
        return kmeans;
    }
}
