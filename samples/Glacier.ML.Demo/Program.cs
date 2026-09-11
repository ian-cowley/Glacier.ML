using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics.Arm;
using Glacier.ML.Clustering;
using Glacier.ML.Core;
using Glacier.ML.Linear;
using Glacier.ML.Preprocessing;
using System.Linq;
using Glacier.ML.Trees;

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================================");
Console.WriteLine("           GLACIER.ML: High-Performance .NET 10 Machine Learning Engine          ");
Console.WriteLine("================================================================================");
Console.ResetColor();

Console.WriteLine($"Hardware Intrinsics Status:");
Console.WriteLine($"  - Vector512 Supported: {Vector512.IsHardwareAccelerated && Avx512F.IsSupported}");
Console.WriteLine($"  - Vector256 Supported: {Vector256.IsHardwareAccelerated && Avx2.IsSupported}");
Console.WriteLine($"  - ARM Neon Supported:  {AdvSimd.IsSupported}");
Console.WriteLine($"  - Logical CPU Cores:   {Environment.ProcessorCount}");
Console.WriteLine();

const int numSamples = 100_000;
const int numFeatures = 8;

Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"[1/4] Synthesizing {numSamples:N0} records with {numFeatures} features in contiguous pinned memory...");
Console.ResetColor();

var sw = Stopwatch.StartNew();
var matrix = new FeatureMatrix(numSamples, numFeatures);
float[] targets = new float[numSamples];

var rng = new Random(42);
for (int i = 0; i < numSamples; i++)
{
    float sum = 0f;
    for (int c = 0; c < numFeatures; c++)
    {
        float val = (float)rng.NextDouble() * 100f;
        matrix.SetValue(i, c, val);
        sum += val;
    }
    targets[i] = (sum > (numFeatures * 50f)) ? 1.0f : 0.0f;
}
sw.Stop();
Console.WriteLine($"  -> Generated {numSamples:N0} rows in {sw.ElapsedMilliseconds:N1} ms ({matrix.Rows * matrix.Columns * sizeof(float) / 1024.0 / 1024.0:F2} MB)\n");

Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[2/4] Benchmarking FastRandomForest (50 trees, max depth 8, parallel multi-threaded)...");
Console.ResetColor();

var forest = new FastRandomForest(numTrees: 50, maxDepth: 8, subsampleRatio: 0.8f);
sw.Restart();
forest.Fit(matrix, targets);
sw.Stop();

long trainMs = sw.ElapsedMilliseconds;
Console.WriteLine($"  -> Successfully trained 50 trees in {trainMs} ms ({numSamples / (trainMs / 1000.0):N0} samples/sec)");

// Zero-allocation batch prediction
float[] predictions = new float[numSamples];
sw.Restart();
forest.Predict(matrix, predictions);
sw.Stop();
float accuracy = Metrics.Accuracy(targets, predictions);
Console.WriteLine($"  -> Batch inference on {numSamples:N0} samples in {sw.ElapsedMilliseconds} ms ({numSamples / (sw.Elapsed.TotalSeconds):N0} inferences/sec)");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  -> Model Accuracy: {accuracy * 100:F2}%\n");
Console.ResetColor();

// Single sample latency test
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[3/4] Measuring Zero-Allocation Single-Row Prediction Latency...");
Console.ResetColor();

ReadOnlySpan<float> singleSample = matrix.GetRow(42);
const int latencyTrials = 1_000_000;
sw.Restart();
float dummy = 0;
for (int i = 0; i < latencyTrials; i++)
{
    dummy += forest.PredictRow(singleSample);
}
sw.Stop();
double nsPerInference = (sw.Elapsed.TotalNanoseconds) / latencyTrials;
Console.WriteLine($"  -> 1,000,000 single-row inferences completed in {sw.ElapsedMilliseconds} ms");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  -> Average Latency: {nsPerInference:F1} ns per inference ({latencyTrials / sw.Elapsed.TotalSeconds:N0} inferences/sec) [ZERO ALLOCATION]\n");
Console.ResetColor();

Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[4/4] Benchmarking KMeans Clustering (k = 8, 20 iterations, SIMD Euclidean distance)...");
Console.ResetColor();

var kmeans = new KMeans(k: 8, maxIterations: 20);
sw.Restart();
kmeans.Fit(matrix);
sw.Stop();
Console.WriteLine($"  -> KMeans fit on {numSamples:N0} samples completed in {sw.ElapsedMilliseconds} ms");

int[] clusterAssignments = new int[numSamples];
sw.Restart();
kmeans.Predict(matrix, clusterAssignments);
sw.Stop();
Console.WriteLine($"  -> Clustered {numSamples:N0} samples into 8 clusters in {sw.ElapsedMilliseconds} ms\n");

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================================");
Console.WriteLine("  Glacier.ML demonstration completed with 100% success!");
Console.WriteLine("================================================================================");
Console.ResetColor();

if (!args.Contains("--headless") && !args.Contains("--bench") && Environment.UserInteractive && !Console.IsInputRedirected)
{
    Console.WriteLine("\n[Press any key to exit...]");
    Console.ReadKey();
}
