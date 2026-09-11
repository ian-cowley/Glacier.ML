using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Glacier.ML.Compute;
using Glacier.ML.Core;

namespace Glacier.ML.Linear;

/// <summary>
/// Hardware-accelerated Logistic Regression classifier trained via Batch Gradient Descent / SGD.
/// Supports bare-metal GPU acceleration (NVIDIA RTX 4060 dGPU, AMD Radeon 890M APU, Auto, and CPU).
/// </summary>
public sealed class FastLogisticRegression : IPredictor
{
    private readonly float _learningRate;
    private readonly int _epochs;
    private readonly float _l2Regularization;
    private readonly GpuTarget _target;
    private float[] _weights = Array.Empty<float>();
    private float _bias;

    public float[] Weights => _weights;
    public float Bias => _bias;
    public float LearningRate => _learningRate;
    public int Epochs => _epochs;
    public float L2Regularization => _l2Regularization;
    public GpuTarget Target => _target;

    public FastLogisticRegression(
        float learningRate = 0.05f,
        int epochs = 50,
        float l2Regularization = 0.001f,
        GpuTarget target = GpuTarget.Auto)
    {
        _learningRate = learningRate;
        _epochs = epochs;
        _l2Regularization = l2Regularization;
        _target = target;
    }

    public void Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        int rows = features.Rows;
        int cols = features.Columns;

        _weights = new float[cols];
        _bias = 0f;

        // For large datasets with GPU available, use Batch Gradient Descent
        if (rows >= 1024 && GpuMlAccelerator.IsGpuAvailable && _target != GpuTarget.Cpu)
        {
            FitBatchGpu(features, targets, rows, cols);
            return;
        }

        // Standard SGD Optimization loop
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            float lr = _learningRate / (1f + 0.01f * epoch); // Learning rate decay

            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> row = features.GetRow(r);
                float y = targets[r] > 0.5f ? 1.0f : 0.0f;

                float z = LinearKernels.DotProduct(row, _weights) + _bias;
                float p = LinearKernels.Sigmoid(z);
                float error = p - y;

                // Update bias
                _bias -= lr * error;

                // Update weights with L2 regularization
                for (int c = 0; c < cols; c++)
                {
                    _weights[c] -= lr * (error * row[c] + _l2Regularization * _weights[c]);
                }
            }
        }
    }

    private void FitBatchGpu(FeatureMatrix features, ReadOnlySpan<float> targets, int rows, int cols)
    {
        float[] z = GC.AllocateArray<float>(rows, pinned: true);
        float[] errors = new float[rows];
        float invRows = 1.0f / rows;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            float lr = _learningRate / (1f + 0.01f * epoch);

            // Forward pass: Z = X * w
            GpuMlAccelerator.ProjectFeatures(features, _weights, z, 1, _target);

            // Compute probabilities & errors
            double sumError = 0;
            float bias = _bias;

            for (int r = 0; r < rows; r++)
            {
                float p = LinearKernels.Sigmoid(z[r] + bias);
                float y = targets[r] > 0.5f ? 1.0f : 0.0f;
                float err = p - y;
                errors[r] = err;
                sumError += err;
            }

            // Update bias
            _bias -= lr * (float)(sumError * invRows);

            // Gradient: grad_w = (1/N) * X^T * error + lambda * w
            float[] gradW = new float[cols];
            float[] fRaw = features.RawArray;

            Parallel.For(0, cols, c =>
            {
                double dot = 0;
                for (int r = 0; r < rows; r++)
                {
                    dot += fRaw[r * cols + c] * errors[r];
                }
                gradW[c] = (float)(dot * invRows) + _l2Regularization * _weights[c];
            });

            // Update weights
            for (int c = 0; c < cols; c++)
            {
                _weights[c] -= lr * gradW[c];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<float> predictions)
    {
        int rows = features.Rows;
        int cols = features.Columns;

        if (rows >= 1024 && GpuMlAccelerator.IsGpuAvailable && _target != GpuTarget.Cpu)
        {
            float[] z = GC.AllocateArray<float>(rows, pinned: true);
            GpuMlAccelerator.ProjectFeatures(features, _weights, z, 1, _target);
            float bias = _bias;
            for (int r = 0; r < rows; r++)
            {
                predictions[r] = (LinearKernels.Sigmoid(z[r] + bias) >= 0.5f) ? 1.0f : 0.0f;
            }
            return;
        }

        for (int r = 0; r < rows; r++)
        {
            predictions[r] = PredictRow(features.GetRow(r));
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
        float z = LinearKernels.DotProduct(row, _weights) + _bias;
        return LinearKernels.Sigmoid(z);
    }
}
