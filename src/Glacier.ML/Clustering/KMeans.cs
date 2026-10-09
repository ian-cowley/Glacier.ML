using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Compute;
using Glacier.ML.Core;

namespace Glacier.ML.Clustering;

/// <summary>
/// Hardware-accelerated k-means clustering engine for .NET 10.
/// Supports bare-metal GPU acceleration (NVIDIA RTX 4060 dGPU, AMD Radeon 890M APU, Auto, and CPU).
/// </summary>
public sealed class KMeans
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
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        _k = k;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _target = target;
    }

    public void Fit(FeatureMatrix features)
    {
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

        float[] minDistances = new float[rows];
        Array.Fill(minDistances, float.MaxValue);

        for (int c = 1; c < k; c++)
        {
            int centroidOffset = (c - 1) * cols;

            if (rows >= 1024)
            {
                Parallel.For(0, rows, r =>
                {
                    float d = KMeansKernels.EuclideanDistanceSquared(features.GetRow(r), _centroids.AsSpan(centroidOffset, cols));
                    if (d < minDistances[r]) minDistances[r] = d;
                });
            }
            else
            {
                ReadOnlySpan<float> lastCentroid = _centroids.AsSpan(centroidOffset, cols);
                for (int r = 0; r < rows; r++)
                {
                    float d = KMeansKernels.EuclideanDistanceSquared(features.GetRow(r), lastCentroid);
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
        float[] newCentroids = new float[k * cols];
        int[] clusterCounts = new int[k];

        int numThreads = Math.Min(Environment.ProcessorCount, Math.Max(1, rows / 512));
        float[][] threadCentroids = new float[numThreads][];
        int[][] threadCounts = new int[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            threadCentroids[t] = new float[k * cols];
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

            // M-step: Recompute centroids in parallel with thread-local buffers
            if (numThreads > 1)
            {
                Parallel.For(0, numThreads, t =>
                {
                    Array.Clear(threadCentroids[t]);
                    Array.Clear(threadCounts[t]);

                    int startRow = t * rows / numThreads;
                    int endRow = (t == numThreads - 1) ? rows : (t + 1) * rows / numThreads;

                    float[] localCentroids = threadCentroids[t];
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
                    float[] tCent = threadCentroids[t];
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

            float maxShift = 0f;
            for (int c = 0; c < k; c++)
            {
                int count = Math.Max(1, clusterCounts[c]);
                int offset = c * cols;
                float shift = 0f;

                for (int d = 0; d < cols; d++)
                {
                    float updated = newCentroids[offset + d] / count;
                    float diff = updated - _centroids[offset + d];
                    shift += diff * diff;
                    _centroids[offset + d] = updated;
                }

                if (shift > maxShift) maxShift = shift;
            }

            if (maxShift < _tolerance)
                break; // Converged
        }

        UpdateCentroidNorms();
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<int> clusterAssignments)
    {
        Predict(features, clusterAssignments, _target);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<int> clusterAssignments, GpuTarget target)
    {
        int rows = features.Rows;
        int cols = features.Columns;
        if (rows == 0 || _centroids.Length == 0)
        {
            return;
        }

        if (_dimensions > 0 && cols != _dimensions)
        {
            throw new ArgumentException($"Expected {_dimensions} features but got {cols}.", nameof(features));
        }

        int k = _dimensions > 0 ? _centroids.Length / _dimensions : _k;
        if (k == 0)
        {
            return;
        }

        if (clusterAssignments.Length < rows)
        {
            throw new ArgumentException("Cluster assignments span must be at least as large as the number of rows.", nameof(clusterAssignments));
        }

        if (features.IsColumnar)
        {
            unsafe
            {
                fixed (int* pAssign = clusterAssignments)
                {
                    nint assignAddr = (nint)pAssign;
                    Parallel.For(0, rows, r =>
                    {
                        int* pOut = (int*)assignAddr;
                        float minD = float.MaxValue;
                        int bestC = 0;
                        for (int c = 0; c < k; c++)
                        {
                            int offset = c * cols;
                            float d = 0f;
                            for (int dim = 0; dim < cols; dim++)
                            {
                                float diff = features.GetColumn(dim)[r] - _centroids[offset + dim];
                                d += diff * diff;
                            }
                            if (d < minD)
                            {
                                minD = d;
                                bestC = c;
                            }
                        }
                        pOut[r] = bestC;
                    });
                }
            }
            return;
        }

        if (GpuMlAccelerator.AssignClustersGpu(features.Span, _centroids, clusterAssignments, rows, k, cols, target))
        {
            return;
        }

        if (_centroidNormsSq.Length != k)
        {
            UpdateCentroidNorms();
        }

        unsafe
        {
            fixed (float* pData = features.Span)
            fixed (float* pCent = _centroids)
            fixed (float* pNorms = _centroidNormsSq)
            fixed (int* pAssign = clusterAssignments)
            {
                if (rows < 512 || Environment.ProcessorCount <= 1)
                {
                    KMeansKernels.PredictBatchDecomposed(pData, pCent, pNorms, 0, rows, cols, k, pAssign);
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
                nint normAddr = (nint)pNorms;
                nint assignAddr = (nint)pAssign;

                Parallel.For(0, numBlocks, b =>
                {
                    float* localData = (float*)dataAddr;
                    float* localCent = (float*)centAddr;
                    float* localNorms = (float*)normAddr;
                    int* localAssign = (int*)assignAddr;

                    int startRow = b * blockSize;
                    int endRow = Math.Min(startRow + blockSize, rows);

                    KMeansKernels.PredictBatchDecomposed(localData, localCent, localNorms, startRow, endRow, cols, k, localAssign);
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

        _centroids = new float[k * cols];
        var rng = new Random(42);

        // k-means++ initialization directly on columnar data
        int firstIdx = rng.Next(0, rows);
        for (int d = 0; d < cols; d++)
        {
            _centroids[d] = features.GetColumn(d)[firstIdx];
        }

        float[] minDistances = new float[rows];
        Array.Fill(minDistances, float.MaxValue);

        for (int c = 1; c < k; c++)
        {
            int centroidOffset = (c - 1) * cols;

            Parallel.For(0, rows, r =>
            {
                float d = 0f;
                for (int dim = 0; dim < cols; dim++)
                {
                    float diff = features.GetColumn(dim)[r] - _centroids[centroidOffset + dim];
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
                _centroids[nextOffset + d] = features.GetColumn(d)[chosenIdx];
            }
        }

        // Iterative optimization loop
        int[] assignments = new int[rows];
        float[] newCentroids = new float[k * cols];
        int[] clusterCounts = new int[k];

        int numThreads = Math.Min(Environment.ProcessorCount, Math.Max(1, rows / 512));
        float[][] threadCentroids = new float[numThreads][];
        int[][] threadCounts = new int[numThreads][];
        for (int t = 0; t < numThreads; t++)
        {
            threadCentroids[t] = new float[k * cols];
            threadCounts[t] = new int[k];
        }

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            Array.Clear(newCentroids);
            Array.Clear(clusterCounts);

            // E-step: Assign points to nearest centroid directly over column spans
            Parallel.For(0, rows, r =>
            {
                float minD = float.MaxValue;
                int bestC = 0;
                for (int c = 0; c < k; c++)
                {
                    int offset = c * cols;
                    float d = 0f;
                    for (int dim = 0; dim < cols; dim++)
                    {
                        float diff = features.GetColumn(dim)[r] - _centroids[offset + dim];
                        d += diff * diff;
                    }
                    if (d < minD)
                    {
                        minD = d;
                        bestC = c;
                    }
                }
                assignments[r] = bestC;
            });

            // M-step: Recompute centroids in parallel with thread-local buffers
            if (numThreads > 1)
            {
                Parallel.For(0, numThreads, t =>
                {
                    Array.Clear(threadCentroids[t]);
                    Array.Clear(threadCounts[t]);

                    int startRow = t * rows / numThreads;
                    int endRow = (t == numThreads - 1) ? rows : (t + 1) * rows / numThreads;

                    float[] localCentroids = threadCentroids[t];
                    int[] localCounts = threadCounts[t];

                    for (int r = startRow; r < endRow; r++)
                    {
                        int cluster = assignments[r];
                        localCounts[cluster]++;
                        int offset = cluster * cols;
                        for (int d = 0; d < cols; d++)
                        {
                            localCentroids[offset + d] += features.GetColumn(d)[r];
                        }
                    }
                });

                for (int t = 0; t < numThreads; t++)
                {
                    float[] tCent = threadCentroids[t];
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
                    for (int d = 0; d < cols; d++)
                    {
                        newCentroids[offset + d] += features.GetColumn(d)[r];
                    }
                }
            }

            float maxShift = 0f;
            for (int c = 0; c < k; c++)
            {
                int count = Math.Max(1, clusterCounts[c]);
                int offset = c * cols;
                float shift = 0f;

                for (int d = 0; d < cols; d++)
                {
                    float updated = newCentroids[offset + d] / count;
                    float diff = updated - _centroids[offset + d];
                    shift += diff * diff;
                    _centroids[offset + d] = updated;
                }

                if (shift > maxShift) maxShift = shift;
            }

            if (maxShift < _tolerance)
                break; // Converged
        }

        UpdateCentroidNorms();
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
            float sum = 0f;
            int offset = c * _dimensions;
            for (int d = 0; d < _dimensions; d++)
            {
                float v = _centroids[offset + d];
                sum += v * v;
            }
            _centroidNormsSq[c] = sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public unsafe int PredictRow(ReadOnlySpan<float> row)
    {
        fixed (float* pRow = row)
        fixed (float* pCent = _centroids)
        {
            return KMeansKernels.FindNearestCentroid(pRow, pCent, _k, _dimensions);
        }
    }
}
