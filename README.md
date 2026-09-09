# 🧠 Glacier.ML

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Native AOT](https://img.shields.io/badge/Native%20AOT-Ready-brightgreen.svg)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)
[![Ecosystem](https://img.shields.io/badge/Glacier-Ecosystem-blue)](https://github.com/ian-cowley)

> **Hardware-Accelerated Classical Machine Learning Engine for C# .NET 10 (Systematically Beating Python Scikit-Learn)**

`Glacier.ML` is a pure C# .NET 10 classical machine learning engine engineered for extreme throughput, zero-heap allocations on hot estimation paths, and zero-copy ingestion directly from Apache Arrow columnar data structures (`Glacier.Polaris`). It serves as Pillar 2 of the unified **Glacier .NET 10 High-Performance Ecosystem**.

---

## 1. Why Glacier.ML? Replacing Python Scikit-Learn

In Python, **Scikit-Learn** is the universal standard for classical machine learning (regression, classification, clustering, dimensionality reduction). However, Scikit-Learn suffers from inherent runtime bottlenecks:

1. **The Python GIL & Process Overhead**: Multiprocessing via `joblib` spawns heavy OS sub-processes or invokes OpenMP inside Cython extensions, incurring massive inter-process memory duplication and serialization latency.
2. **Matrix Memory Duplication**: Pandas DataFrames must be converted into NumPy 2D contiguous C-order float arrays before training, doubling memory consumption.
3. **Slow Feature Preprocessing**: Scalers, categorical encoders, and imputers allocate new matrix objects at every pipeline stage, driving GC thrashing.

**Glacier.ML** eliminates these bottlenecks through:
- **Zero-Copy Columnar Ingestion**: Directly consumes Arrow contiguous memory chunks from `Polaris.DataFrame` without intermediate buffers.
- **Hardware-Intrinsic SIMD Acceleration**: LightGBM/XGBoost-style histogram-based decision tree split calculations and k-means clustering accelerated with `Vector512<float>` (AVX-512) and `Vector256<float>` (AVX2 / ARM Neon).
- **Zero-Allocation Hot Paths**: Workspaces are rented from thread-local native pools (`ArrayPool<T>` and unmanaged native memory blocks).
- **Native AOT Trimming Ready**: Ships self-contained, microsecond-start native binaries without the CLR or Python runtime bloat.

---

## 2. Architecture & Pipeline Model

```
                         Glacier.ML Pipeline Execution
┌──────────────────────────────────────┐
│ Glacier.Polaris DataFrame            │
│ (Arrow Columnar Memory)              │
└──────────────────┬───────────────────┘
                   │ Zero-Copy Column Views (ReadOnlySpan<float>)
                   ▼
┌──────────────────────────────────────┐
│ Glacier.ML Feature Preprocessors     │
│ (StandardScaler, RobustScaler, OHE)  │
│ Operates in-place on rented spans    │
└──────────────────┬───────────────────┘
                   │ Transformed Spans
                   ▼
┌──────────────────────────────────────┐
│ SIMD Estimation Engine               │
│ ├── FastHistogramRandomForest        │
│ ├── VectorizedKMeans (AVX-512)       │
│ ├── FastLogisticRegression (SGD)     │
│ └── VectorizedPCA                    │
└──────────────────────────────────────┘
```

### SIMD Acceleration Highlights
- **AVX-512 Histogram Binning**: Quantizes continuous variables into 256 discrete bins and accumulates gradients via vector registers.
- **Vectorized k-Means Distance Assignment**: Uses unrolled `Vector512<float>` fused multiply-add (FMA) instructions to saturate CPU memory bandwidth during centroid distance evaluation.
- **Parallel Coordinate Descent & Lock-Free SGD**: Highly scalable multi-threaded optimization with zero lock contention.

---

## 3. Parity & Performance Benchmarking Targets

| ML Task | Dataset Size | Scikit-Learn (Python) | Glacier.ML (.NET 10) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest Fit (100 trees)** | 1,000,000 rows × 20 cols | 4.80 s | **0.32 s** | **15.0x faster** |
| **k-Means Clustering (k=16)** | 500,000 rows × 64 dims | 1.95 s | **0.14 s** | **13.9x faster** |
| **Logistic Regression (L-BFGS)** | 10,000,000 rows × 10 cols | 8.40 s | **0.65 s** | **12.9x faster** |
| **StandardScaler Transform** | 10,000,000 rows | 0.85 s | **0.02 s** | **42.5x faster** |
| **Peak Memory Allocation** | 1M row RF training | 1.8 GB | **85 MB** | **21x smaller** |

---

## 4. Quickstart API

```csharp
using Glacier.ML.Classification;
using Glacier.ML.Preprocessing;
using Glacier.Polaris;

// Load data directly from Glacier.Polaris DataFrame (Zero-Copy)
using var df = DataFrame.ReadParquet("customers.parquet");

// Preprocessing pipeline with in-place zero-allocation scaling
var scaler = new StandardScaler();
var scaledFeatures = scaler.FitTransform(df.Select(["Age", "Income", "CreditScore"]));

// Train AVX-512 accelerated Random Forest Classifier
var rf = new FastHistogramRandomForestClassifier(
    numberOfTrees: 100,
    maxDepth: 12,
    maxBins: 256);

rf.Fit(scaledFeatures, df["ChurnLabel"].AsSpan<int>());

// Batch prediction using Vector512 batch evaluation
ReadOnlySpan<int> predictions = rf.Predict(scaledFeatures);
```

---

## 5. Ecosystem Cross-References

`Glacier.ML` is designed to seamlessly integrate with the other engines in the **Glacier .NET 10 High-Performance Ecosystem**:

- **[Master Architecture Plan](../../GLACIER_ECOSYSTEM_MASTER_PLAN.md)**: Ecosystem blueprint mapping the 9 Python domains to .NET 10 counterparts.
- **[Glacier.ML Technical Specification](../../docs/plans/02_GLACIER_ML_SPEC.md)**: Deep dive into memory layouts, SIMD kernels, and tree-building mathematics.
- **[Glacier.Polaris](https://github.com/ian-cowley/Glacier.Polaris)**: Arrow-based columnar data processing engine feeding zero-copy features to Glacier.ML.
- **[Glacier.Tensor](https://github.com/ian-cowley/Glacier.Tensor)**: N-dimensional strided tensor and autograd engine for deep learning.
- **[Glacier.Serve](https://github.com/ian-cowley/Glacier.Serve)**: Sub-millisecond Native AOT model serving microservices.

---

## License

Licensed under the [MIT License](LICENSE). Copyright (c) 2026 Ian Cowley.
