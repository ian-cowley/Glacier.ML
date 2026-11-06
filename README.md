![Glacier.ML Banner](assets/banner.jpg)

# 🧠 Glacier.ML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Native AOT](https://img.shields.io/badge/Native%20AOT-Ready-brightgreen.svg)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)
[![NuGet Version](https://img.shields.io/nuget/v/Glacier.ML.svg)](https://www.nuget.org/packages/Glacier.ML/)
[![Ecosystem](https://img.shields.io/badge/Glacier-Ecosystem-blue)](https://github.com/ian-cowley)
[![Website](https://img.shields.io/badge/Website-glaciersoftware.dev-66fcf1.svg)](https://glaciersoftware.dev/)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-orange.svg?logo=buy-me-a-coffee)](https://buymeacoffee.com/glaciersoftware)

> **Hardware-Accelerated Classical Machine Learning Engine for C# .NET 10**

`Glacier.ML` is a pure C# .NET 10 classical machine learning engine engineered for extreme throughput, zero-heap allocations on hot estimation paths, and zero-copy ingestion directly from Apache Arrow columnar data structures (`Glacier.Polaris`). It serves as Pillar 2 of the unified **Glacier .NET 10 High-Performance Ecosystem**.

---

## 🚀 Key Highlights

- **Hardware-Accelerated Kernels**: Saturated vectorization using `Vector512<float>`, `Vector256<float>`, `Vector128<float>`, and `AdvSimd` for dot products, squared Euclidean distances, and vector reductions.
- **Bare-Metal GPU Acceleration**: Direct P/Invoke CUDA Driver execution (`nvcuda.dll`) offloading K-Means cluster assignment and persistent ring-buffer vector kernels (`VectorAdd` / `VectorFma`) to NVIDIA GPUs (`sm_75+`) without CUDA SDK runtime dependencies, with automatic SIMD CPU fallback.
- **Dynamic Hardware Target Scaling**: Select between `GpuTarget.Auto`, `GpuTarget.Nvidia`, and `GpuTarget.Cpu` dynamically based on batch sizes and hardware availability.
- **4-Way Unrolled Histogram Splitting**: Breaks CPU write-port read-after-write (RAW) dependency hazards with stack-allocated sub-histograms residing entirely in L1d cache.
- **Pure Zero-Allocation Inference**: `PredictRow(ReadOnlySpan<float>)` executes in **< 360 nanoseconds** without touching the managed heap or triggering garbage collection.
- **Multi-Core Parallelism**: Trains Random Forest ensembles and assigns KMeans clusters across all logical CPU cores and GPU streaming multiprocessors without GIL bottlenecks.
- **Zero-Copy Columnar Interop**: Directly train and preprocess `Glacier.Polaris` DataFrames and writable columnar `FeatureMatrix` buffers without data duplication.
- **Native AOT Ready**: 100% compatible with Ahead-of-Time compilation (`IsAotCompatible=true`) for microsecond cold starts and single-file native binaries.

---

## 📊 Performance Benchmarks

*Benchmarked on .NET 10.0 (x64 AVX-512 24 logical cores vs. NVIDIA GeForce RTX 4060 Laptop GPU sm_89)*

| Operation | Dataset / Configuration | Scikit-Learn (Python) | Glacier.ML (CPU) | Glacier.ML (Bare-Metal GPU) | Speedup vs Python |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KMeans Batch Predict** | 100,000 samples × 8 features | ~42 ms | 5.6 ms | **< 1.0 ms (0.5 ms)** | **> 80x** |
| **KMeans Batch Predict (Fleet)** | 50,000 samples × 10 features | 0.80 ms (62.3M/s) | **0.46 ms (109.9M/s)** | — | **1.76x faster** |
| **KMeans Fit** | 100,000 rows × 8 features ($k=8$, 20 iters) | ~680 ms | 180 ms | **127 ms** | **5.3x** |
| **FastLinearRegression Inference** | 100,000 samples | ~35 ms | 6 ms | **1 ms** | **35x** |
| **FastPCA Covariance ($X^T X$)** | 100,000 samples × 64 dims | ~120 ms | 35 ms | **14 ms** | **8.5x** |
| **FastPCA 3D Projection** | 100,000 samples | ~45 ms | 18 ms | **7 ms** | **6.4x** |
| **Random Forest Fit** | 100,000 rows, 50 trees, depth 8 | ~14,200 ms | **3,603 ms** | — | **3.9x** |
| **Single-Sample Inference** | Zero-alloc `PredictRow` | ~150,000 ns | **352 ns** | — | **426x** |

---

## 🛠️ Architecture Overview

```mermaid
graph TD
    A[Polaris DataFrame / Arrow RecordBatch] -->|Zero-Copy Pointers| B[FeatureMatrix Contiguous / Columnar Memory]
    B --> C[Preprocessing: StandardScaler / MinMaxScaler / OneHot]
    C --> D[SIMD & Bare-Metal CUDA Kernels]
    D --> E1[FastRandomForest / FastDecisionTree: 4-Way L1d Histograms]
    D --> E2[KMeans: Vector512 / CUDA Distance Sinks]
    D --> E3[FastLogisticRegression: FMA Vectorized Dot Products]
    E1 --> F[Sub-Microsecond Zero-Allocation Predict]
    E2 --> F
    E3 --> F
```

---

## 💻 Quick Start

### 1. Training a Multi-Threaded Random Forest
```csharp
using Glacier.ML.Core;
using Glacier.ML.Trees;

// Initialize contiguous pinned feature matrix
using var features = new FeatureMatrix(100_000, 8);
float[] targets = LoadLabels();

// Train 50 trees concurrently across all CPU cores
var forest = new FastRandomForest(numTrees: 50, maxDepth: 8);
forest.Fit(features, targets);

// Zero-allocation single-sample inference (< 400 ns)
ReadOnlySpan<float> sample = features.GetRow(0);
float prediction = forest.PredictRow(sample);
float probability = forest.PredictProbability(sample);
```

### 2. Bare-Metal GPU KMeans Clustering
```csharp
using Glacier.ML.Clustering;
using Glacier.ML.Core;

// Automatically selects NVIDIA GPU or AVX-512 CPU
var kmeans = new KMeans(k: 16, maxIterations: 20, target: GpuTarget.Auto);
kmeans.Fit(features);

int[] assignments = new int[features.Rows];
// Predict on 100,000 samples in < 1 ms via bare-metal GPU kernel
kmeans.Predict(features, assignments);
```

### 3. Polaris DataFrame Direct Integration
```csharp
using Glacier.ML.Interop;
using Glacier.Polaris;

// Directly fit models on Polaris DataFrames
var forest = df.FitRandomForest(
    targetColumn: "churn", 
    featureColumns: new[] { "age", "balance", "tenure", "score" },
    numTrees: 100
);

var kmeans = df.FitKMeans(
    featureColumns: new[] { "x", "y", "z" }, 
    k: 4
);
```

### 4. Persistent Ring Buffer Megakernel Dispatch
```csharp
using Glacier.ML.Compute;
using Glacier.ML.Core;

// Dispatches VectorAdd or VectorFma directly to the persistent GPU ring buffer worker when available,
// bypassing per-call cuLaunchKernel launch latency with automatic AVX-512 / AVX2 fallback
GpuMlAccelerator.VectorAdd(aSpan, bSpan, outSpan, GpuTarget.Auto);
GpuMlAccelerator.VectorFma(aSpan, bSpan, outSpan, scalar: -0.05f, GpuTarget.Auto);
```

---

## 🧪 Testing & Verification

Run the comprehensive unit test suite:
```bash
dotnet test tests/Glacier.ML.Tests/Glacier.ML.Tests.csproj -c Release
```

Run the live interactive benchmark demo:
```bash
dotnet run --project samples/Glacier.ML.Demo/Glacier.ML.Demo.csproj -c Release
```

---

## 🌐 Ecosystem Cross-References

`Glacier.ML` seamlessly connects across the Glacier High-Performance Computing Ecosystem:
- **[Glacier.Polaris](https://github.com/ian-cowley/PolarsPlus)**: Columnar data engine providing zero-copy features to Glacier.ML.
- **[Glacier.Gpu](https://github.com/ian-cowley/Glacier.Gpu)**: Bare-metal CUDA/HIP driver bindings and persistent ring-buffer megakernels.
- **[Glacier.Tensor](https://github.com/ian-cowley/Glacier.Tensor)**: Strided N-D tensors and autograd for deep learning.
- **[Glacier.Serve](https://github.com/ian-cowley/Glacier.Serve)**: Sub-millisecond Native AOT model serving microservices.

---

## 🆕 What's New in v1.1.0

- **True Zero-Copy Writable Columnar Transforms** — `StandardScaler`, `MinMaxScaler`, and `OneHotEncoder` now transform writable columnar `FeatureMatrix(Memory<float>[])` instances directly in-place and guard read-only columnar views against silent mutation loss.
- **Hardened CUDA Stream & Persistent Ring Buffer Lifecycle** — Per-thread CUDA context binding on `GpuMlStreamContext`, lockstep PTX `.version` / `.target` normalization via `Glacier.Gpu 1.1.0`, and barrier-safe persistent ring buffer host-mapped memory management.
- **Strict Native AOT & Bounds Hygiene** — Full `IsAotCompatible` + `TreatWarningsAsErrors` verification, hardened kernel span bounds checks, and `Glacier.Polaris 1.1.0` + `Glacier.Gpu 1.1.0` integration.

---

## Credits

Developed by Ian Cowley and Antigravity (Google DeepMind).

---

## 📜 License
Licensed under the [MIT License](LICENSE). Copyright (c) 2026 Ian Cowley.

---

## ❄️ Glacier Ecosystem & Support

Part of the **[Glacier Ecosystem](https://glaciersoftware.dev/)** — high-performance, zero-allocation developer tooling and data engines natively built for modern C# and .NET 10.

* 🌐 **Website & Documentation**: [glaciersoftware.dev](https://glaciersoftware.dev/)
* ☕ **Buy Me a Coffee**: [buymeacoffee.com/glaciersoftware](https://buymeacoffee.com/glaciersoftware)
* ❤️ **GitHub Sponsors**: [github.com/sponsors/ian-cowley](https://github.com/sponsors/ian-cowley)
* 🐙 **All Repositories**: [github.com/ian-cowley](https://github.com/ian-cowley)
