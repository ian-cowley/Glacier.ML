using System;
using System.Linq;
using Glacier.ML.Clustering;
using Glacier.ML.Core;
using Glacier.ML.Decomposition;
using Glacier.ML.Trees;
using Glacier.Polaris;
using Glacier.Polaris.Data;

namespace Glacier.ML.Interop;

public static class PolarisExtensions
{
    /// <summary>
    /// Converts selected numeric columns of a Polaris DataFrame into a zero-copy aligned FeatureMatrix.
    /// Ingests Polaris Series columnar memory directly without row-by-row transposition copies.
    /// </summary>
    public static FeatureMatrix ToFeatureMatrix(this DataFrame df, params string[] featureColumnNames)
    {
        if (df == null) throw new ArgumentNullException(nameof(df));
        if (featureColumnNames == null || featureColumnNames.Length == 0)
            throw new ArgumentException("At least one feature column name must be provided.", nameof(featureColumnNames));

        int rows = df.RowCount;
        int cols = featureColumnNames.Length;
        var columnMemories = new ReadOnlyMemory<float>[cols];

        for (int c = 0; c < cols; c++)
        {
            string colName = featureColumnNames[c];
            var series = df.Columns.FirstOrDefault(x => x.Name.Equals(colName, StringComparison.OrdinalIgnoreCase))
                ?? throw new ArgumentException($"Column '{colName}' not found in DataFrame.");

            if (series is Series<float> floatSeries)
            {
                // ZERO-COPY: directly wrap Polaris columnar memory
                columnMemories[c] = floatSeries.Memory;
            }
            else if (series is Series<double> doubleSeries)
            {
                float[] conv = new float[rows];
                var span = doubleSeries.Memory.Span;
                for (int r = 0; r < rows; r++) conv[r] = (float)span[r];
                columnMemories[c] = conv;
            }
            else if (series is Series<int> intSeries)
            {
                float[] conv = new float[rows];
                var span = intSeries.Memory.Span;
                for (int r = 0; r < rows; r++) conv[r] = (float)span[r];
                columnMemories[c] = conv;
            }
            else if (series is Series<long> longSeries)
            {
                float[] conv = new float[rows];
                var span = longSeries.Memory.Span;
                for (int r = 0; r < rows; r++) conv[r] = (float)span[r];
                columnMemories[c] = conv;
            }
            else
            {
                throw new NotSupportedException($"Series type '{series.GetType().Name}' is not supported for FeatureMatrix conversion.");
            }
        }

        return new FeatureMatrix(columnMemories, featureColumnNames);
    }

    /// <summary>
    /// Fits a FastRandomForest model directly from a Polaris DataFrame with zero data copying.
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

        var features = df.ToFeatureMatrix(featureColumns);
        var forest = new FastRandomForest(numTrees, maxDepth);

        if (targetSeries is Series<float> ft)
        {
            // ZERO-COPY: pass Float32 target span directly without array copy
            forest.Fit(features, ft.Memory.Span);
            return forest;
        }

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
        else if (targetSeries is Series<long> lt)
        {
            var span = lt.Memory.Span;
            for (int r = 0; r < rows; r++) targets[r] = (float)span[r];
        }
        else
        {
            throw new NotSupportedException($"Target series type '{targetSeries.GetType().Name}' is not supported.");
        }

        forest.Fit(features, targets);
        return forest;
    }

    /// <summary>
    /// Fits a KMeans clustering model directly from a Polaris DataFrame with zero data copying.
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

    /// <summary>
    /// Fits a FastPCA dimensionality reduction model directly from a Polaris DataFrame with zero data copying.
    /// </summary>
    public static FastPCA FitPCA(
        this DataFrame df, 
        int nComponents, 
        params string[] featureColumns)
    {
        var features = df.ToFeatureMatrix(featureColumns);
        var pca = new FastPCA(nComponents);
        pca.Fit(features);
        return pca;
    }

    /// <summary>
    /// Fits a FastPCA dimensionality reduction model directly from a Polaris DataFrame with specified target.
    /// </summary>
    public static FastPCA FitPCA(
        this DataFrame df, 
        string[] featureColumns, 
        int nComponents = 2, 
        GpuTarget target = GpuTarget.Auto)
    {
        var features = df.ToFeatureMatrix(featureColumns);
        var pca = new FastPCA(nComponents, target);
        pca.Fit(features);
        return pca;
    }

    /// <summary>
    /// Fits a FastPCA dimensionality reduction model directly from a Polaris DataFrame.
    /// </summary>
    public static FastPCA FitPCA(
        this DataFrame df, 
        int nComponents, 
        GpuTarget target, 
        params string[] featureColumns)
    {
        var features = df.ToFeatureMatrix(featureColumns);
        var pca = new FastPCA(nComponents, target);
        pca.Fit(features);
        return pca;
    }
}
