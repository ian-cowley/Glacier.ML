using System;
using System.Diagnostics;
using Glacier.ML.Clustering;
using Glacier.ML.Core;
using Xunit;

namespace Glacier.ML.Tests
{
    public class AdversarialKMeansTests
    {
        /// <summary>
        /// CHALLENGE 1: Batch size boundary testing.
        /// When batch size N < K (e.g. N = 1, 2, 3 with K = 5), Predict currently clamps
        /// k = Math.Min(_k, rows), causing Predict to ignore clusters >= N and assign incorrect clusters.
        /// </summary>
        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(10)]
        [InlineData(64)]
        [InlineData(511)]
        [InlineData(512)]
        [InlineData(1024)]
        [InlineData(10000)]
        public void Adversarial_KMeans_Predict_Matches_PredictRow_AcrossBatchSizes(int numTestSamples)
        {
            const int trainRows = 500;
            const int cols = 4;
            const int k = 5;

            var trainMatrix = new FeatureMatrix(trainRows, cols);
            var rng = new Random(42);
            for (int r = 0; r < trainRows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    trainMatrix.SetValue(r, c, (float)rng.NextDouble() * 50f);
                }
            }

            var kmeans = new KMeans(k: k, maxIterations: 15);
            kmeans.Fit(trainMatrix);

            var testMatrix = new FeatureMatrix(numTestSamples, cols);
            for (int r = 0; r < numTestSamples; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    testMatrix.SetValue(r, c, (float)rng.NextDouble() * 50f);
                }
            }

            int[] batchAssignments = new int[numTestSamples];
            kmeans.Predict(testMatrix, batchAssignments);

            // Compare against individual PredictRow (scalar reference oracle)
            for (int r = 0; r < numTestSamples; r++)
            {
                int expected = kmeans.PredictRow(testMatrix.GetRow(r));
                Assert.Equal(expected, batchAssignments[r]);
            }
        }

        /// <summary>
        /// CHALLENGE 1b: Targeted minimal reproduction of the N < K bug.
        /// Fits 4 separated clusters, then queries 1 sample near cluster 3.
        /// If k = Math.Min(_k, rows) is evaluated, k becomes 1, and only cluster 0 is tested.
        /// </summary>
        [Fact]
        public void Adversarial_KMeans_SmallBatch_UnderK_SingleSampleAssignsToNearestCluster()
        {
            const int cols = 2;
            const int k = 4;
            const int ptsPerCluster = 50;
            var trainMatrix = new FeatureMatrix(ptsPerCluster * k, cols);

            for (int c = 0; c < k; c++)
            {
                float center = c * 20f;
                for (int i = 0; i < ptsPerCluster; i++)
                {
                    int row = c * ptsPerCluster + i;
                    trainMatrix.SetValue(row, 0, center + (i % 2 == 0 ? 0.1f : -0.1f));
                    trainMatrix.SetValue(row, 1, center + (i % 2 == 0 ? 0.1f : -0.1f));
                }
            }

            var kmeans = new KMeans(k: k, maxIterations: 30);
            kmeans.Fit(trainMatrix);

            // Test single sample located near cluster 3 (center = 60, 60)
            var singleSample = new FeatureMatrix(1, cols);
            singleSample.SetValue(0, 0, 60.1f);
            singleSample.SetValue(0, 1, 59.9f);

            int[] assignment = new int[1];
            kmeans.Predict(singleSample, assignment);

            int expected = kmeans.PredictRow(singleSample.GetRow(0));
            // Fails with Expected: 3, Actual: 0 due to k = Math.Min(_k, rows)
            Assert.Equal(expected, assignment[0]);
        }

        /// <summary>
        /// CHALLENGE 2: Floating-point scale & catastrophic cancellation stress.
        /// Matrix decomposition ||c||^2 - 2<x, c> works with 100% precision up to scale ~1,000.
        /// Beyond 10,000 in float32, catastrophic cancellation in ||c||^2 - 2<x, c> introduces errors.
        /// </summary>
        [Theory]
        [InlineData(10.0f, true)]        // Normal scale: 100% equivalent
        [InlineData(1000.0f, true)]      // Moderate scale: 100% equivalent
        [InlineData(10000.0f, false)]    // 10^4 scale: precision loss (~6% divergence)
        [InlineData(100000.0f, false)]   // 10^5 scale: catastrophic cancellation (~71% divergence)
        public void Adversarial_KMeans_ExtremeFloatingPoint_LargeCoordinates(float scale, bool expectPerfectMatch)
        {
            const int numSamples = 200;
            const int dim = 8;
            const int k = 5;

            var rng = new Random(1001);
            float[] centroids = new float[k * dim];
            for (int i = 0; i < centroids.Length; i++)
            {
                centroids[i] = scale + (float)rng.NextDouble() * 50f;
            }

            float[] centroidNormsSq = new float[k];
            for (int c = 0; c < k; c++)
            {
                float sum = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float v = centroids[c * dim + d];
                    sum += v * v;
                }
                centroidNormsSq[c] = sum;
            }

            float[] data = new float[numSamples * dim];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = scale + (float)rng.NextDouble() * 50f;
            }

            int[] expectedAssignments = new int[numSamples];
            int[] actualAssignments = new int[numSamples];

            for (int r = 0; r < numSamples; r++)
            {
                expectedAssignments[r] = KMeansKernels.FindNearestCentroid(
                    data.AsSpan(r * dim, dim),
                    centroids,
                    k,
                    dim);
            }

            KMeansKernels.PredictBatchDecomposed(
                data,
                centroids,
                centroidNormsSq,
                0,
                numSamples,
                dim,
                k,
                actualAssignments);

            int matches = 0;
            for (int r = 0; r < numSamples; r++)
            {
                if (expectedAssignments[r] == actualAssignments[r]) matches++;
            }

            if (expectPerfectMatch)
            {
                Assert.Equal(numSamples, matches);
            }
            else
            {
                // Documents empirical observation of float32 cancellation at large coordinates
                double matchRate = (double)matches / numSamples;
                Assert.True(matchRate < 1.0, $"Expected precision loss at scale {scale}, but got 100% match");
            }
        }

        [Fact]
        public void Adversarial_KMeans_ExtremeFloatingPoint_NegativeCoordinates()
        {
            const int numSamples = 300;
            const int dim = 6;
            const int k = 4;

            var rng = new Random(2002);
            float[] centroids = new float[k * dim];
            for (int i = 0; i < centroids.Length; i++)
            {
                centroids[i] = -5000f + (float)rng.NextDouble() * 200f;
            }

            float[] centroidNormsSq = new float[k];
            for (int c = 0; c < k; c++)
            {
                float sum = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float v = centroids[c * dim + d];
                    sum += v * v;
                }
                centroidNormsSq[c] = sum;
            }

            float[] data = new float[numSamples * dim];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = -5000f + (float)rng.NextDouble() * 200f;
            }

            int[] expected = new int[numSamples];
            int[] actual = new int[numSamples];

            for (int r = 0; r < numSamples; r++)
            {
                expected[r] = KMeansKernels.FindNearestCentroid(data.AsSpan(r * dim, dim), centroids, k, dim);
            }

            KMeansKernels.PredictBatchDecomposed(data, centroids, centroidNormsSq, 0, numSamples, dim, k, actual);

            for (int r = 0; r < numSamples; r++)
            {
                Assert.Equal(expected[r], actual[r]);
            }
        }

        [Fact]
        public void Adversarial_KMeans_ExtremeFloatingPoint_TinySeparations()
        {
            const int numSamples = 200;
            const int dim = 4;
            const int k = 3;

            var rng = new Random(3003);
            float[] centroids = new float[k * dim];
            for (int c = 0; c < k; c++)
            {
                for (int d = 0; d < dim; d++)
                {
                    centroids[c * dim + d] = 1.0f + (c * 0.001f);
                }
            }

            float[] centroidNormsSq = new float[k];
            for (int c = 0; c < k; c++)
            {
                float sum = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float v = centroids[c * dim + d];
                    sum += v * v;
                }
                centroidNormsSq[c] = sum;
            }

            float[] data = new float[numSamples * dim];
            for (int r = 0; r < numSamples; r++)
            {
                int targetC = r % k;
                for (int d = 0; d < dim; d++)
                {
                    data[r * dim + d] = centroids[targetC * dim + d] + ((float)rng.NextDouble() - 0.5f) * 0.0001f;
                }
            }

            int[] expected = new int[numSamples];
            int[] actual = new int[numSamples];

            for (int r = 0; r < numSamples; r++)
            {
                expected[r] = KMeansKernels.FindNearestCentroid(data.AsSpan(r * dim, dim), centroids, k, dim);
            }

            KMeansKernels.PredictBatchDecomposed(data, centroids, centroidNormsSq, 0, numSamples, dim, k, actual);

            for (int r = 0; r < numSamples; r++)
            {
                Assert.Equal(expected[r], actual[r]);
            }
        }

        [Theory]
        [InlineData(2, 5, 500)]
        [InlineData(3, 8, 500)]
        [InlineData(4, 4, 1000)]
        [InlineData(8, 10, 1000)]
        [InlineData(16, 8, 1000)]
        [InlineData(32, 16, 1000)]
        public void Adversarial_KMeans_Mathematical_OrdinalEquivalence_Fuzz(int dim, int k, int numSamples)
        {
            var rng = new Random(4004 + dim);

            float[] centroids = new float[k * dim];
            for (int i = 0; i < centroids.Length; i++)
            {
                centroids[i] = (float)rng.NextDouble() * 100f;
            }

            float[] centroidNormsSq = new float[k];
            for (int c = 0; c < k; c++)
            {
                float sum = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float v = centroids[c * dim + d];
                    sum += v * v;
                }
                centroidNormsSq[c] = sum;
            }

            float[] data = new float[numSamples * dim];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)rng.NextDouble() * 100f;
            }

            int[] directEuclidean = new int[numSamples];
            int[] decomposedFormula = new int[numSamples];

            for (int r = 0; r < numSamples; r++)
            {
                directEuclidean[r] = KMeansKernels.FindNearestCentroid(data.AsSpan(r * dim, dim), centroids, k, dim);
            }

            KMeansKernels.PredictBatchDecomposed(data, centroids, centroidNormsSq, 0, numSamples, dim, k, decomposedFormula);

            for (int r = 0; r < numSamples; r++)
            {
                Assert.Equal(directEuclidean[r], decomposedFormula[r]);
            }
        }
    }
}
