using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using Glacier.ML.Core;

namespace Glacier.ML.Linear;

/// <summary>
/// SIMD-accelerated binary Logistic Regression classifier trained via Batch Gradient Descent / SGD
/// on multi-core CPU (<see cref="GpuTarget.Auto"/> and <see cref="GpuTarget.Cpu"/>).
/// </summary>
public sealed class FastLogisticRegression : IPredictor, IEstimator<FastLogisticRegression>
{
    private readonly float _learningRate;
    private readonly int _epochs;
    private readonly float _l2Regularization;
    private readonly GpuTarget _target;
    private float[] _weights = Array.Empty<float>();
    private float _bias;
    private int _nFeatures;

    public float[] Weights => _weights;
    public float Bias => _bias;
    public float LearningRate => _learningRate;
    public int Epochs => _epochs;
    public float L2Regularization => _l2Regularization;
    public GpuTarget Target => _target;
    public int NFeatures => _nFeatures;

    public FastLogisticRegression(
        float learningRate = 0.05f,
        int epochs = 50,
        float l2Regularization = 0.001f,
        GpuTarget target = GpuTarget.Auto)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "LearningRate must be finite and positive.");
        if (epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
        if (!float.IsFinite(l2Regularization) || l2Regularization < 0f)
            throw new ArgumentOutOfRangeException(nameof(l2Regularization), "L2Regularization must be finite and non-negative.");

        _learningRate = learningRate;
        _epochs = epochs;
        _l2Regularization = l2Regularization;
        _target = target;
    }

    FastLogisticRegression IEstimator<FastLogisticRegression>.Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        Fit(features, targets);
        return this;
    }

    private void ValidateTarget()
    {
        if (_target is GpuTarget.Nvidia or GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GpuTarget.{_target} is not supported by FastLogisticRegression (uses CPU SIMD). Use Auto or Cpu.");
    }

    public void Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        ArgumentNullException.ThrowIfNull(features);
        ValidateTarget();

        int rows = features.Rows;
        int cols = features.Columns;
        if (rows <= 0)
            throw new ArgumentException("FeatureMatrix must contain at least one row.", nameof(features));
        if (targets.Length < rows)
            throw new ArgumentException("Targets length must match or exceed feature rows.", nameof(targets));

        _nFeatures = cols;
        _weights = new float[cols];
        _bias = 0f;

        if (rows >= 1024)
        {
            FitBatchCpuSimd(features, targets, rows, cols);
            return;
        }

        // Standard SGD Optimization loop with zero per-row heap allocation
        bool isColumnar = features.IsColumnar;
        float[]? rentedRow = isColumnar && cols > 256 ? ArrayPool<float>.Shared.Rent(cols) : null;
        try
        {
            Span<float> rowScratch = isColumnar
                ? (cols <= 256 ? stackalloc float[cols] : rentedRow.AsSpan(0, cols))
                : Span<float>.Empty;
            ReadOnlySpan<float> rawSpan = isColumnar ? ReadOnlySpan<float>.Empty : features.Span;

            for (int epoch = 0; epoch < _epochs; epoch++)
            {
                float lr = _learningRate / (1f + 0.01f * epoch);
                float decay = 1f - lr * _l2Regularization;

                for (int r = 0; r < rows; r++)
                {
                    if (isColumnar)
                    {
                        features.CopyRowTo(r, rowScratch);
                    }
                    ReadOnlySpan<float> row = isColumnar
                        ? (ReadOnlySpan<float>)rowScratch
                        : rawSpan.Slice((int)((long)r * cols), cols);

                    float y = targets[r] > 0.5f ? 1.0f : 0.0f;
                    float z = LinearKernels.DotProduct(row, _weights) + _bias;
                    float p = LinearKernels.Sigmoid(z);
                    float error = p - y;

                    _bias -= lr * error;

                    float negLrErr = -lr * error;
                    for (int c = 0; c < cols; c++)
                    {
                        _weights[c] = _weights[c] * decay + negLrErr * row[c];
                    }
                }
            }
        }
        finally
        {
            if (rentedRow != null) ArrayPool<float>.Shared.Return(rentedRow);
        }
    }

    private unsafe void FitBatchCpuSimd(FeatureMatrix features, ReadOnlySpan<float> targets, int rows, int cols)
    {
        float[] z = new float[rows];
        float[] errors = new float[rows];
        float[] gradW = new float[cols];
        float invRows = 1.0f / rows;
        bool isColumnar = features.IsColumnar;

        ReadOnlyMemory<float>[]? colMemories = null;
        if (isColumnar)
        {
            colMemories = new ReadOnlyMemory<float>[cols];
            for (int c = 0; c < cols; c++) colMemories[c] = features.GetColumnMemory(c);
        }

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            float lr = _learningRate / (1f + 0.01f * epoch);

            if (isColumnar)
            {
                Array.Fill(z, _bias);
                for (int c = 0; c < cols; c++)
                {
                    AccumulateScaledColumnSimd(colMemories![c].Span, _weights[c], z);
                }
            }
            else
            {
                fixed (float* pData = features.Span)
                fixed (float* pWeights = _weights)
                fixed (float* pZ = z)
                {
                    nint dataAddr = (nint)pData;
                    nint weightsAddr = (nint)pWeights;
                    nint zAddr = (nint)pZ;
                    float bias = _bias;
                    Parallel.For(0, rows, r =>
                    {
                        var row = new ReadOnlySpan<float>((float*)dataAddr + ((long)r * cols), cols);
                        var w = new ReadOnlySpan<float>((float*)weightsAddr, cols);
                        ((float*)zAddr)[r] = LinearKernels.DotProduct(row, w) + bias;
                    });
                }
            }

            double sumError = 0.0;
            for (int r = 0; r < rows; r++)
            {
                float p = LinearKernels.Sigmoid(z[r]);
                float y = targets[r] > 0.5f ? 1.0f : 0.0f;
                float err = p - y;
                errors[r] = err;
                sumError += err;
            }

            _bias -= lr * (float)(sumError * invRows);

            if (isColumnar)
            {
                Parallel.For(0, cols, c =>
                {
                    float dot = LinearKernels.DotProduct(colMemories![c].Span, errors);
                    gradW[c] = dot * invRows + _l2Regularization * _weights[c];
                });
            }
            else
            {
                fixed (float* pData = features.Span)
                fixed (float* pErrors = errors)
                {
                    nint dataAddr = (nint)pData;
                    nint errAddr = (nint)pErrors;
                    Parallel.For(0, cols, c =>
                    {
                        float* pRaw = (float*)dataAddr;
                        float* pE = (float*)errAddr;
                        double dot = 0.0;
                        for (int r = 0; r < rows; r++)
                        {
                            dot += pRaw[(long)r * cols + c] * pE[r];
                        }
                        gradW[c] = (float)(dot * invRows) + _l2Regularization * _weights[c];
                    });
                }
            }

            for (int c = 0; c < cols; c++)
            {
                _weights[c] -= lr * gradW[c];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<float> predictions)
    {
        ArgumentNullException.ThrowIfNull(features);
        ValidateTarget();
        if (_weights.Length == 0 || _nFeatures <= 0)
            throw new InvalidOperationException("FastLogisticRegression must be fit before calling Predict.");

        int rows = features.Rows;
        int cols = features.Columns;
        if (cols != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {cols}.", nameof(features));
        if (predictions.Length < rows)
            throw new ArgumentException("Predictions span is smaller than feature rows.", nameof(predictions));

        if (rows == 0) return;

        if (features.IsColumnar)
        {
            Span<float> logits = predictions.Slice(0, rows);
            logits.Fill(_bias);
            for (int c = 0; c < cols; c++)
            {
                AccumulateScaledColumnSimd(features.GetColumn(c), _weights[c], logits);
            }
            for (int r = 0; r < rows; r++)
            {
                logits[r] = logits[r] >= 0f ? 1.0f : 0.0f;
            }
            return;
        }

        ReadOnlySpan<float> data = features.Span;
        for (int r = 0; r < rows; r++)
        {
            predictions[r] = PredictRow(data.Slice((int)((long)r * cols), cols));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictRow(ReadOnlySpan<float> row)
    {
        return PredictProbability(row) >= 0.5f ? 1.0f : 0.0f;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float PredictProbability(ReadOnlySpan<float> row)
    {
        if (_weights.Length == 0 || _nFeatures <= 0)
            throw new InvalidOperationException("FastLogisticRegression must be fit before calling PredictProbability.");
        if (row.Length != _nFeatures)
            throw new ArgumentException($"Expected {_nFeatures} features but got {row.Length}.", nameof(row));

        float z = LinearKernels.DotProduct(row, _weights) + _bias;
        return LinearKernels.Sigmoid(z);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AccumulateScaledColumnSimd(ReadOnlySpan<float> col, float weight, Span<float> dst)
    {
        int length = col.Length;
        int i = 0;
        ref float rCol = ref MemoryMarshal.GetReference(col);
        ref float rDst = ref MemoryMarshal.GetReference(dst);

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var vw = Vector512.Create(weight);
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vc = Vector512.LoadUnsafe(ref rCol, (nuint)i);
                var vd = Vector512.LoadUnsafe(ref rDst, (nuint)i);
                Vector512.FusedMultiplyAdd(vc, vw, vd).StoreUnsafe(ref rDst, (nuint)i);
                i += step;
            }
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var vw = Vector256.Create(weight);
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vc = Vector256.LoadUnsafe(ref rCol, (nuint)i);
                var vd = Vector256.LoadUnsafe(ref rDst, (nuint)i);
                Vector256.FusedMultiplyAdd(vc, vw, vd).StoreUnsafe(ref rDst, (nuint)i);
                i += step;
            }
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            var vw = Vector128.Create(weight);
            int step = Vector128<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var vc = Vector128.LoadUnsafe(ref rCol, (nuint)i);
                var vd = Vector128.LoadUnsafe(ref rDst, (nuint)i);
                Vector128.FusedMultiplyAdd(vc, vw, vd).StoreUnsafe(ref rDst, (nuint)i);
                i += step;
            }
        }

        for (; i < length; i++)
        {
            dst[i] += col[i] * weight;
        }
    }
}
