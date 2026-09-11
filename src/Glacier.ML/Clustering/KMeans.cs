using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Compute;
using Glacier.ML.Core;
using Glacier.Tensor.Compute;

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
    private int _dimensions;

    public int K => _k;
    public int MaxIterations => _maxIterations;
    public float Tolerance => _tolerance;
    public GpuTarget Target => _target;
    public float[] Centroids => _centroids;
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
            ReadOnlySpan<float> lastCentroid = _centroids.AsSpan((c - 1) * cols, cols);
            double sumDist = 0;

            for (int r = 0; r < rows; r++)
            {
                float d = KMeansKernels.EuclideanDistanceSquared(features.GetRow(r), lastCentroid);
                if (d < minDistances[r]) minDistances[r] = d;
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

            // M-step: Recompute centroids
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
        int k = Math.Min(_k, rows);

        if (GpuMlAccelerator.AssignClustersGpu(features.Span, _centroids, clusterAssignments, rows, k, cols, target))
        {
            return;
        }

        unsafe
        {
            fixed (float* pData = features.RawArray)
            fixed (float* pCent = _centroids)
            fixed (int* pAssign = clusterAssignments)
            {
                for (int r = 0; r < rows; r++)
                {
                    float* pRow = pData + (r * cols);
                    pAssign[r] = KMeansKernels.FindNearestCentroid(pRow, pCent, k, cols);
                }
            }
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
