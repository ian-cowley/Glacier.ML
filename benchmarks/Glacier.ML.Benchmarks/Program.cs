using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;
using Glacier.ML.Clustering;
using Glacier.ML.Core;

namespace Glacier.ML.Benchmarks;

public class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0].Equals("--bdn", StringComparison.OrdinalIgnoreCase))
        {
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
            return;
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("================================================================================");
        Console.WriteLine("  GLACIER.ML K-MEANS EMPIRICAL BENCHMARK (AVX-512 + MULTI-THREADED M-STEP)      ");
        Console.WriteLine("================================================================================");
        Console.ResetColor();

        // 1. 50,000 samples, 8 dimensions, 8 clusters (10 iterations)
        RunKMeansTest(rows: 50_000, cols: 8, k: 8, iters: 10, "50k samples, D=8, K=8");

        // 2. 1,000,000 samples, 8 dimensions, 8 clusters (10 iterations)
        RunKMeansTest(rows: 1_000_000, cols: 8, k: 8, iters: 10, "1M samples, D=8, K=8");

        // 3. 500,000 samples, Spatial 2D (D=2, K=4, 10 iterations)
        RunKMeansTest(rows: 500_000, cols: 2, k: 4, iters: 10, "Spatial 2D: 500k samples, D=2, K=4");

        // 4. 500,000 samples, Spatial 3D (D=3, K=8, 10 iterations)
        RunKMeansTest(rows: 500_000, cols: 3, k: 8, iters: 10, "Spatial 3D: 500k samples, D=3, K=8");
    }

    private static void RunKMeansTest(int rows, int cols, int k, int iters, string label)
    {
        Console.WriteLine($"\n--- Benchmark: {label} ---");
        var matrix = new FeatureMatrix(rows, cols);
        var rng = new Random(42);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                matrix.SetValue(r, c, (float)rng.NextDouble() * 100f);
            }
        }

        // Warmup
        var warmupKmeans = new KMeans(k: k, maxIterations: 2);
        warmupKmeans.Fit(matrix);

        // Timed run
        var sw = Stopwatch.StartNew();
        var kmeans = new KMeans(k: k, maxIterations: iters);
        kmeans.Fit(matrix);
        sw.Stop();

        double totalMs = sw.Elapsed.TotalMilliseconds;
        double msPerIter = totalMs / iters;
        double samplesPerSec = (double)rows * iters / (totalMs / 1000.0);

        Console.WriteLine($"  Total Time ({iters} iters): {totalMs:F2} ms");
        Console.WriteLine($"  Time per Iteration:       {msPerIter:F2} ms");
        Console.WriteLine($"  Throughput:               {samplesPerSec:N0} samples/sec");
    }
}
