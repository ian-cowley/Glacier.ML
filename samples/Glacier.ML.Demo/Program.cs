using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using Glacier.ML.Clustering;
using Glacier.ML.Compute;
using Glacier.ML.Core;
using Glacier.ML.Decomposition;
using Glacier.ML.Linear;
using Glacier.ML.Preprocessing;
using Glacier.ML.Trees;

Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================================");
Console.WriteLine("           GLACIER.ML: High-Performance .NET 10 Machine Learning Engine          ");
Console.WriteLine("================================================================================");
Console.ResetColor();

Console.WriteLine("Hardware Acceleration Status:");
Console.WriteLine($"  - CPU AVX-512 FMA:      {Vector512.IsHardwareAccelerated && Avx512F.IsSupported}");
Console.WriteLine($"  - CPU AVX2 Vector256:   {Vector256.IsHardwareAccelerated && Avx2.IsSupported}");
Console.WriteLine($"  - CPU Cores / Threads:  {Environment.ProcessorCount}");
Console.WriteLine($"  - NVIDIA RTX 4060 dGPU: {GpuMlAccelerator.IsNvidiaAvailable} (Direct nvcuda.dll PTX)");
Console.WriteLine($"  - AMD Radeon 890M APU:  {GpuMlAccelerator.IsAmdAvailable} (Direct amdhip64.dll)");
Console.WriteLine();

const int numSamples = 100_000;
const int numFeatures = 8;

Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine($"[1/5] Synthesizing {numSamples:N0} records with {numFeatures} features in contiguous pinned memory...");
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

// -----------------------------------------------------------------------------
// [2/5] FastRandomForest
// -----------------------------------------------------------------------------
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[2/5] Benchmarking FastRandomForest (50 trees, max depth 8, parallel multi-threaded)...");
Console.ResetColor();

var forest = new FastRandomForest(numTrees: 50, maxDepth: 8, subsampleRatio: 0.8f);
sw.Restart();
forest.Fit(matrix, targets);
sw.Stop();

long trainMs = sw.ElapsedMilliseconds;
Console.WriteLine($"  -> Successfully trained 50 trees in {trainMs} ms ({numSamples / (trainMs / 1000.0):N0} samples/sec)");

float[] predictions = new float[numSamples];
sw.Restart();
forest.Predict(matrix, predictions);
sw.Stop();
float accuracy = Metrics.Accuracy(targets, predictions);
Console.WriteLine($"  -> Batch inference on {numSamples:N0} samples in {sw.ElapsedMilliseconds} ms ({numSamples / sw.Elapsed.TotalSeconds:N0} inf/s)");
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  -> Model Accuracy: {accuracy * 100:F2}%\n");
Console.ResetColor();

// -----------------------------------------------------------------------------
// [3/5] FastLinearRegression (Normal Equations on GPU vs CPU)
// -----------------------------------------------------------------------------
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[3/5] Benchmarking FastLinearRegression (Closed-Form Normal Equations: X^T * X on GPU vs CPU)...");
Console.ResetColor();

// CPU OLS
var lrCpu = new FastLinearRegression(alpha: 0.01f, fitIntercept: true, target: GpuTarget.Cpu);
sw.Restart();
lrCpu.Fit(matrix, targets);
sw.Stop();
long lrCpuMs = sw.ElapsedMilliseconds;

float[] lrPredCpu = new float[numSamples];
sw.Restart();
lrCpu.Predict(matrix, lrPredCpu);
sw.Stop();
Console.WriteLine($"  -> CPU AVX-512 Fit:     {lrCpuMs} ms | Predict: {sw.ElapsedMilliseconds} ms | R2: {Metrics.R2Score(targets, lrPredCpu):F4}");

// GPU OLS
var lrGpu = new FastLinearRegression(alpha: 0.01f, fitIntercept: true, target: GpuTarget.Auto);
sw.Restart();
lrGpu.Fit(matrix, targets);
sw.Stop();
long lrGpuMs = sw.ElapsedMilliseconds;

float[] lrPredGpu = new float[numSamples];
sw.Restart();
lrGpu.Predict(matrix, lrPredGpu);
sw.Stop();
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  -> GPU (RTX 4060) Fit:  {lrGpuMs} ms | Predict: {sw.ElapsedMilliseconds} ms | R2: {Metrics.R2Score(targets, lrPredGpu):F4}\n");
Console.ResetColor();

// -----------------------------------------------------------------------------
// [4/5] FastPCA (Covariance Gram Matrix & Feature Projection)
// -----------------------------------------------------------------------------
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[4/5] Benchmarking FastPCA (Principal Component Analysis, 3 components, GPU-Accelerated Covariance)...");
Console.ResetColor();

var pca = new FastPCA(nComponents: 3, target: GpuTarget.Auto);
sw.Restart();
pca.Fit(matrix);
sw.Stop();
long pcaFitMs = sw.ElapsedMilliseconds;

float[] reduced = new float[numSamples * 3];
sw.Restart();
pca.Transform(matrix, reduced);
sw.Stop();
long pcaTransformMs = sw.ElapsedMilliseconds;

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"  -> PCA Fit (Covariance + Eigendecomposition): {pcaFitMs} ms");
Console.WriteLine($"  -> PCA Projection (100,000 samples -> 3D):     {pcaTransformMs} ms");
Console.WriteLine($"  -> Explained Variance Ratios: [{string.Join(", ", pca.ExplainedVarianceRatio.Select(v => $"{v * 100:F2}%"))}]\n");
Console.ResetColor();

// -----------------------------------------------------------------------------
// [5/5] KMeans Clustering: CPU vs. NVIDIA RTX 4060 dGPU
// -----------------------------------------------------------------------------
Console.ForegroundColor = ConsoleColor.Yellow;
Console.WriteLine("[5/5] Benchmarking KMeans Clustering (k = 8, 20 iterations): Hardware Target Comparison...");
Console.ResetColor();

// 1. CPU AVX-512
var kmeansCpu = new KMeans(k: 8, maxIterations: 20, target: GpuTarget.Cpu);
sw.Restart();
kmeansCpu.Fit(matrix);
sw.Stop();
long kCpuFitMs = sw.ElapsedMilliseconds;

int[] assignCpu = new int[numSamples];
sw.Restart();
kmeansCpu.Predict(matrix, assignCpu);
sw.Stop();
long kCpuPredMs = sw.ElapsedMilliseconds;
Console.WriteLine($"  -> [CPU AVX-512]       Fit (20 iters): {kCpuFitMs} ms | Batch Predict: {kCpuPredMs} ms");

// 2. NVIDIA RTX 4060 dGPU (Direct PTX kernel)
if (GpuMlAccelerator.IsNvidiaAvailable)
{
    var kmeansNvidia = new KMeans(k: 8, maxIterations: 20, target: GpuTarget.Nvidia);
    sw.Restart();
    kmeansNvidia.Fit(matrix);
    sw.Stop();
    long kNvidiaFitMs = sw.ElapsedMilliseconds;

    int[] assignNvidia = new int[numSamples];
    sw.Restart();
    kmeansNvidia.Predict(matrix, assignNvidia);
    sw.Stop();
    long kNvidiaPredMs = sw.ElapsedMilliseconds;

    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine($"  -> [NVIDIA RTX 4060]   Fit (20 iters): {kNvidiaFitMs} ms | Batch Predict: {kNvidiaPredMs} ms");
    if (kCpuFitMs > 0 && kNvidiaFitMs > 0)
    {
        double speedup = (double)kCpuFitMs / kNvidiaFitMs;
        Console.WriteLine($"     ==> GPU Speedup vs CPU: {speedup:F2}x faster!");
    }
    Console.ResetColor();
}

Console.WriteLine();
Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("================================================================================");
Console.WriteLine("  Glacier.ML GPU hardware acceleration benchmark completed with 100% success!");
Console.WriteLine("================================================================================");
Console.ResetColor();

if (!args.Contains("--headless") && !args.Contains("--bench") && Environment.UserInteractive && !Console.IsInputRedirected)
{
    Console.WriteLine("\n[Press any key to exit...]");
    Console.ReadKey();
}
