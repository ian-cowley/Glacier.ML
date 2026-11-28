using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Compute;
using Glacier.ML.Core;

namespace Glacier.ML.Clustering;

/// <summary>
/// Hardware-accelerated k-means clustering engine for .NET 10.
/// Supports bare-metal NVIDIA CUDA acceleration (<see cref="GpuTarget.Nvidia"/>, <see cref="GpuTarget.Auto"/>) and multi-core SIMD CPU (<see cref="GpuTarget.Cpu"/>).
/// </summary>
public sealed class KMeans : IUnsupervisedEstimator<KMeans>
{
    private readonly int _k;
    private readonly int _maxIterations;
    private readonly float _tolerance;
    private readonly GpuTarget _target;
    private float[] _centroids = Array.Empty<float>();
    private float[] _centroidNormsSq = Array.Empty<float>();
    private int _dimensions;

    public int K => _k;
    public int MaxIterations => _maxIterations;
    public float Tolerance => _tolerance;
    public GpuTarget Target => _target;
    public float[] Centroids => _centroids;
    public float[] CentroidNormsSq => _centroidNormsSq;
    public int Dimensions => _dimensions;

    public KMeans(int k = 8, int maxIterations = 100, float tolerance = 1e-4f, GpuTarget target = GpuTarget.Auto)
    {
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Number of clusters k must be positive.");
        if (maxIterations <= 0) throw new ArgumentOutOfRangeException(nameof(maxIterations), "MaxIterations must be positive.");
        if (!float.IsFinite(tolerance) || tolerance < 0f)
            throw new ArgumentOutOfRangeException(nameof(tolerance), "Tolerance must be finite and non-negative.");

        _k = k;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _target = target;
    }

    KMeans IUnsupervisedEstimator<KMeans>.Fit(FeatureMatrix features)
    {
        Fit(features);
        return this;
    }

    public void Fit(FeatureMatrix features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (_target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{_target} is not supported by KMeans. Use Auto, Nvidia, or Cpu.");
        if (features.Rows <= 0)
            throw new ArgumentException("FeatureMatrix must contain at least one row.", nameof(features));

        if (features.IsColumnar)
        {
            FitColumnar(features);
            return;
        }

        int rows = features.Rows;
        int cols = features.Columns;
        _dimensions = cols;
        int k = Math.Min(_k, rows);

        _centroids = new float[k * cols];
        var rng = new Random(42);

        // k-means++ initialization
        // 1. Choose first centroid uniformly at random
        int firstIdx = rng.Next(0, rows);
        features.GetRow(firstIdx).CopyTo(_centroids.AsSpan(0, cols));

        double[] minDistances = new double[rows];
        Array.Fill(minDistances, double.MaxValue);

        for (int c = 1; c < k; c++)
        {
            int centroidOffset = (c - 1) * cols;

            if (rows >= 1024)
            {
                Parallel.For(0, rows, r =>
                {
                    double d = KMeansKernels.EuclideanDistanceSquared(features.GetRow(r), _centroids.AsSpan(centroidOffset, cols));
                    if (d < minDistances[r]) minDistances[r] = d;
                });
            }
            else
            {
                ReadOnlySpan<float> lastCentroid = _centroids.AsSpan(centroidOffset, cols);
                for (int r = 0; r < rows; r++)
                {
                    double d = KMeansKernels.EuclideanDistanceSquared(features.GetRow(r), lastCentroid);
                    if (d < minDistances[r]) minDistances[r] = d;
                }
            }

            double sumDist = 0;
            for (int r = 0; r < rows; r++)
            {
                sumDist += minDistances[r];
            }

            // Choose next centroid proportional to distance squared
            double target = rng.NextDouble() * sumDist;
            double cumulative = 0;
            int chosenIdx = rows - 1;

            for (int r = 0; r < rows; r++)
            {
                cumulative += minDistances[r];
                if (cumulative >= target)
                {
                    chosenIdx = r;
                    break;
                }
            }

            features.GetRow(chosenIdx).CopyTo(_centroids.AsSpan(c * cols, cols));
        }

        // Iterative optimization loop
        int[] assignments = new int[rows];
        double[] newCentroids = new double[k * cols];
        int[] clusterCounts = new int[k];

        int numThreads = Math.Min(Environment.ProcessorCount, Math.Max(1, rows / 512));
        double[][] threadCentroids = new double[numThreads][];
        int[][] threadCounts = new int[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            threadCentroids[t] = new double[k * cols];
            threadCounts[t] = new int[k];
        }

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Array.Clear(newCentroids);
            Array.Clear(clusterCounts);

            // E-step: Assign points to nearest centroid using GPU or multi-core AVX-512
            bool usedGpu = GpuMlAccelerator.AssignClustersGpu(features.Span, _centroids, assignments, rows, k, cols, _target);
            if (!usedGpu)
            {
                Parallel.For(0, rows, r =>
                {
                    assignments[r] = KMeansKernels.FindNearestCentroid(features.GetRow(r), _centroids, k, cols);
                });
            }

            // M-step: Recompute centroids in parallel with thread-local double accumulators
            if (numThreads > 1)
            {
                Parallel.For(0, numThreads, t =>
                {
                    Array.Clear(threadCentroids[t]);
                    Array.Clear(threadCounts[t]);

                    int startRow = t * rows / numThreads;
                    int endRow = (t == numThreads - 1) ? rows : (t + 1) * rows / numThreads;

                    double[] localCentroids = threadCentroids[t];
                    int[] localCounts = threadCounts[t];

                    for (int r = startRow; r < endRow; r++)
                    {
                        int cluster = assignments[r];
                        localCounts[cluster]++;
                        int offset = cluster * cols;
                        ReadOnlySpan<float> row = features.GetRow(r);
                        for (int d = 0; d < cols; d++)
                        {
                            localCentroids[offset + d] += row[d];
                        }
                    }
                });

                for (int t = 0; t < numThreads; t++)
                {
                    double[] tCent = threadCentroids[t];
                    int[] tCnt = threadCounts[t];
                    for (int c = 0; c < k; c++)
                    {
                        clusterCounts[c] += tCnt[c];
                    }
                    for (int i = 0; i < k * cols; i++)
                    {
                        newCentroids[i] += tCent[i];
                    }
                }
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    int cluster = assignments[r];
                    clusterCounts[cluster]++;
                    int offset = cluster * cols;
                    ReadOnlySpan<float> row = features.GetRow(r);
                    for (int d = 0; d < cols; d++)
                    {
                        newCentroids[offset + d] += row[d];
                    }
                }
            }

            double maxShift = 0.0;
            bool hasEmptyCluster = false;
            for (int c = 0; c < k; c++)
            {
                int count = clusterCounts[c];
                if (count == 0)
                {
                    hasEmptyCluster = true;
                    continue;
                }

                int offset = c * cols;
                double shift = 0.0;

                for (int d = 0; d < cols; d++)
                {
                    float updated = (float)(newCentroids[offset + d] / count);
                    double diff = (double)updated - _centroids[offset + d];
                    shift += diff * diff;
                    _centroids[offset + d] = updated;
                }

                if (shift > maxShift) maxShift = shift;
            }

            if (hasEmptyCluster)
            {
                double reseedShift = RelocateEmptyClustersRowMajor(features, assignments, clusterCounts, k, rows, cols);
                if (reseedShift > maxShift) maxShift = reseedShift;
            }

            if (maxShift < _tolerance)
                break; // Converged
        }

        UpdateCentroidNorms();
    }

    private double RelocateEmptyClustersRowMajor(
        FeatureMatrix features,
        int[] assignments,
        int[] clusterCounts,
        int k,
        int rows,
        int cols)
    {
        double maxShift = 0.0;
        bool[] claimed = new bool[rows];

        for (int c = 0; c < k; c++)
        {
            if (clusterCounts[c] > 0) continue;

            int bestRow = -1;
            double maxErr = 0.0;
            for (int r = 0; r < rows; r++)
            {
                if (claimed[r]) continue;
                int assignedCluster = assignments[r];
                double err = KMeansKernels.EuclideanDistanceSquared(
                    features.GetRow(r),
                    _centroids.AsSpan(assignedCluster * cols, cols));
                if (err > maxErr)
                {
                    maxErr = err;
                    bestRow = r;
                }
            }

            if (bestRow >= 0 && maxErr > 0.0)
            {
                claimed[bestRow] = true;
                int offset = c * cols;
                ReadOnlySpan<float> sample = features.GetRow(bestRow);
                double shift = 0.0;
                for (int d = 0; d < cols; d++)
                {
                    float updated = sample[d];
                    double diff = (double)updated - _centroids[offset + d];
                    shift += diff * diff;
                    _centroids[offset + d] = updated;
                }
                if (shift > maxShift) maxShift = shift;
            }
            // Otherwise preserve existing centroid without resetting to origin
        }

        return maxShift;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<int> clusterAssignments)
    {
        Predict(features, clusterAssignments, _target);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<int> clusterAssignments, GpuTarget target)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{target} is not supported by KMeans. Use Auto, Nvidia, or Cpu.");
        if (_dimensions <= 0 || _centroids.Length == 0)
            throw new InvalidOperationException("KMeans must be fit before calling Predict.");

        int rows = features.Rows;
        int cols = features.Columns;
        if (rows == 0)
        {
            return;
        }

        if (cols != _dimensions)
        {
            throw new ArgumentException($"Expected {_dimensions} features but got {cols}.", nameof(features));
        }

        int k = _centroids.Length / _dimensions;
        if (k == 0)
        {
            throw new InvalidOperationException("KMeans has no fitted centroids.");
        }

        if (clusterAssignments.Length < rows)
        {
            throw new ArgumentException("Cluster assignments span must be at least as large as the number of rows.", nameof(clusterAssignments));
        }

        if (features.IsColumnar)
        {
            ReadOnlyMemory<float>[] colMemories = new ReadOnlyMemory<float>[cols];
            for (int d = 0; d < cols; d++)
            {
                colMemories[d] = features.GetColumnMemory(d);
            }

            unsafe
            {
                fixed (int* pAssign = clusterAssignments)
                fixed (float* pCent = _centroids)
                {
                    nint assignAddr = (nint)pAssign;
                    nint centAddr = (nint)pCent;
                    Parallel.For(0, rows, r =>
                    {
                        float[]? rented = cols > 256 ? ArrayPool<float>.Shared.Rent(cols) : null;
                        try
                        {
                            Span<float> rowBuf = cols <= 256 ? stackalloc float[cols] : rented.AsSpan(0, cols);
                            for (int d = 0; d < cols; d++)
                            {
                                rowBuf[d] = colMemories[d].Span[r];
                            }
                            fixed (float* pRow = rowBuf)
                            {
                                ((int*)assignAddr)[r] = KMeansKernels.FindNearestCentroid(pRow, (float*)centAddr, k, cols);
                            }
                        }
                        finally
                        {
                            if (rented != null) ArrayPool<float>.Shared.Return(rented);
                        }
                    });
                }
            }
            return;
        }

        if (GpuMlAccelerator.AssignClustersGpu(features.Span, _centroids, clusterAssignments, rows, k, cols, target))
        {
            return;
        }

        unsafe
        {
            fixed (float* pData = features.Span)
            fixed (float* pCent = _centroids)
            fixed (int* pAssign = clusterAssignments)
            {
                if (rows < 512 || Environment.ProcessorCount <= 1)
                {
                    KMeansKernels.PredictBatchExact(pData, pCent, 0, rows, cols, k, pAssign);
                    return;
                }

                int blockSize = rows switch
                {
                    < 2048 => 512,
                    _ => 1024
                };
                int numBlocks = (rows + blockSize - 1) / blockSize;

                nint dataAddr = (nint)pData;
                nint centAddr = (nint)pCent;
                nint assignAddr = (nint)pAssign;

                Parallel.For(0, numBlocks, b =>
                {
                    float* localData = (float*)dataAddr;
                    float* localCent = (float*)centAddr;
                    int* localAssign = (int*)assignAddr;

                    int startRow = b * blockSize;
                    int endRow = Math.Min(startRow + blockSize, rows);

                    KMeansKernels.PredictBatchExact(localData, localCent, startRow, endRow, cols, k, localAssign);
                });
            }
        }
    }

    private void FitColumnar(FeatureMatrix features)
    {
        int rows = features.Rows;
        int cols = features.Columns;
        _dimensions = cols;
        int k = Math.Min(_k, rows);

        ReadOnlyMemory<float>[] colMemories = new ReadOnlyMemory<float>[cols];
        for (int d = 0; d < cols; d++)
        {
            colMemories[d] = features.GetColumnMemory(d);
        }

        _centroids = new float[k * cols];
        var rng = new Random(42);

        // k-means++ initialization directly on columnar data (zero RawArray copy)
        int firstIdx = rng.Next(0, rows);
        for (int d = 0; d < cols; d++)
        {
            _centroids[d] = colMemories[d].Span[firstIdx];
        }

        double[] minDistances = new double[rows];
        Array.Fill(minDistances, double.MaxValue);

        for (int c = 1; c < k; c++)
        {
            int centroidOffset = (c - 1) * cols;

            Parallel.For(0, rows, r =>
            {
                double d = 0.0;
                for (int dim = 0; dim < cols; dim++)
                {
                    double diff = (double)colMemories[dim].Span[r] - _centroids[centroidOffset + dim];
                    d += diff * diff;
                }
                if (d < minDistances[r]) minDistances[r] = d;
            });

            double sumDist = 0;
            for (int r = 0; r < rows; r++)
            {
                sumDist += minDistances[r];
            }

            double target = rng.NextDouble() * sumDist;
            double cumulative = 0;
            int chosenIdx = rows - 1;

            for (int r = 0; r < rows; r++)
            {
                cumulative += minDistances[r];
                if (cumulative >= target)
                {
                    chosenIdx = r;
                    break;
                }
            }

            int nextOffset = c * cols;
            for (int d = 0; d < cols; d++)
            {
                _centroids[nextOffset + d] = colMemories[d].Span[chosenIdx];
            }
        }

        // Iterative optimization loop
        int[] assignments = new int[rows];
        double[] newCentroids = new double[k * cols];
        int[] clusterCounts = new int[k];

        int numThreads = Math.Min(Environment.ProcessorCount, Math.Max(1, rows / 512));
        double[][] threadCentroids = new double[numThreads][];
        int[][] threadCounts = new int[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            threadCentroids[t] = new double[k * cols];
            threadCounts[t] = new int[k];
        }

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Array.Clear(newCentroids);
            Array.Clear(clusterCounts);

            // E-step: Assign points to nearest centroid directly over hoisted column spans using SIMD
            unsafe
            {
                fixed (float* pCent = _centroids)
                {
                    nint centAddr = (nint)pCent;
                    Parallel.For(0, rows, r =>
                    {
                        float[]? rented = cols > 256 ? ArrayPool<float>.Shared.Rent(cols) : null;
                        try
                        {
                            Span<float> rowBuf = cols <= 256 ? stackalloc float[cols] : rented.AsSpan(0, cols);
                            for (int d = 0; d < cols; d++)
                            {
                                rowBuf[d] = colMemories[d].Span[r];
                            }
                            fixed (float* pRow = rowBuf)
                            {
                                assignments[r] = KMeansKernels.FindNearestCentroid(pRow, (float*)centAddr, k, cols);
                            }
                        }
                        finally
                        {
                            if (rented != null) ArrayPool<float>.Shared.Return(rented);
                        }
                    });
                }
            }

            // M-step: Recompute centroids in parallel with thread-local double buffers (column-outer for cache locality)
            if (numThreads > 1)
            {
                Parallel.For(0, numThreads, t =>
                {
                    Array.Clear(threadCentroids[t]);
                    Array.Clear(threadCounts[t]);

                    int startRow = t * rows / numThreads;
                    int endRow = (t == numThreads - 1) ? rows : (t + 1) * rows / numThreads;

                    double[] localCentroids = threadCentroids[t];
                    int[] localCounts = threadCounts[t];

                    for (int r = startRow; r < endRow; r++)
                    {
                        localCounts[assignments[r]]++;
                    }

                    for (int d = 0; d < cols; d++)
                    {
                        ReadOnlySpan<float> colSpan = colMemories[d].Span;
                        for (int r = startRow; r < endRow; r++)
                        {
                            localCentroids[assignments[r] * cols + d] += colSpan[r];
                        }
                    }
                });

                for (int t = 0; t < numThreads; t++)
                {
                    double[] tCent = threadCentroids[t];
                    int[] tCnt = threadCounts[t];
                    for (int c = 0; c < k; c++)
                    {
                        clusterCounts[c] += tCnt[c];
                    }
                    for (int i = 0; i < k * cols; i++)
                    {
                        newCentroids[i] += tCent[i];
                    }
                }
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    clusterCounts[assignments[r]]++;
                }

                for (int d = 0; d < cols; d++)
                {
                    ReadOnlySpan<float> colSpan = colMemories[d].Span;
                    for (int r = 0; r < rows; r++)
                    {
                        newCentroids[assignments[r] * cols + d] += colSpan[r];
                    }
                }
            }

            double maxShift = 0.0;
            bool hasEmptyCluster = false;
            for (int c = 0; c < k; c++)
            {
                int count = clusterCounts[c];
                if (count == 0)
                {
                    hasEmptyCluster = true;
                    continue;
                }

                int offset = c * cols;
                double shift = 0.0;

                for (int d = 0; d < cols; d++)
                {
                    float updated = (float)(newCentroids[offset + d] / count);
                    double diff = (double)updated - _centroids[offset + d];
                    shift += diff * diff;
                    _centroids[offset + d] = updated;
                }

                if (shift > maxShift) maxShift = shift;
            }

            if (hasEmptyCluster)
            {
                double reseedShift = RelocateEmptyClustersColumnar(colMemories, assignments, clusterCounts, k, rows, cols);
                if (reseedShift > maxShift) maxShift = reseedShift;
            }

            if (maxShift < _tolerance)
                break; // Converged
        }

        UpdateCentroidNorms();
    }

    private double RelocateEmptyClustersColumnar(
        ReadOnlyMemory<float>[] colMemories,
        int[] assignments,
        int[] clusterCounts,
        int k,
        int rows,
        int cols)
    {
        double maxShift = 0.0;
        bool[] claimed = new bool[rows];

        for (int c = 0; c < k; c++)
        {
            if (clusterCounts[c] > 0) continue;

            int bestRow = -1;
            double maxErr = 0.0;
            for (int r = 0; r < rows; r++)
            {
                if (claimed[r]) continue;
                int centOffset = assignments[r] * cols;
                double err = 0.0;
                for (int d = 0; d < cols; d++)
                {
                    double diff = (double)colMemories[d].Span[r] - _centroids[centOffset + d];
                    err += diff * diff;
                }
                if (err > maxErr)
                {
                    maxErr = err;
                    bestRow = r;
                }
            }

            if (bestRow >= 0 && maxErr > 0.0)
            {
                claimed[bestRow] = true;
                int offset = c * cols;
                double shift = 0.0;
                for (int d = 0; d < cols; d++)
                {
                    float updated = colMemories[d].Span[bestRow];
                    double diff = (double)updated - _centroids[offset + d];
                    shift += diff * diff;
                    _centroids[offset + d] = updated;
                }
                if (shift > maxShift) maxShift = shift;
            }
            // Otherwise preserve existing centroid without resetting to origin
        }

        return maxShift;
    }

    private void UpdateCentroidNorms()
    {
        if (_dimensions <= 0 || _centroids.Length == 0) return;

        int k = Math.Min(_k, _centroids.Length / _dimensions);
        if (_centroidNormsSq.Length != k)
        {
            _centroidNormsSq = new float[k];
        }

        for (int c = 0; c < k; c++)
        {
            double sum = 0.0;
            int offset = c * _dimensions;
            for (int d = 0; d < _dimensions; d++)
            {
                double v = _centroids[offset + d];
                sum += v * v;
            }
            _centroidNormsSq[c] = (float)sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe int PredictRow(ReadOnlySpan<float> row)
    {
        if (_dimensions <= 0 || _centroids.Length == 0)
            throw new InvalidOperationException("KMeans must be fit before calling PredictRow.");
        if (row.Length != _dimensions)
            throw new ArgumentException($"Expected {_dimensions} features but got {row.Length}.", nameof(row));

        int actualK = _centroids.Length / _dimensions;
        fixed (float* pRow = row)
        fixed (float* pCent = _centroids)
        {
            return KMeansKernels.FindNearestCentroid(pRow, pCent, actualK, _dimensions);
        }
    }
}
