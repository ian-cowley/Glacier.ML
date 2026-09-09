using System;
using System.Runtime.CompilerServices;
using Glacier.ML.Core;

namespace Glacier.ML.Linear;

/// <summary>
/// Hardware-accelerated Logistic Regression classifier trained via Stochastic Gradient Descent (SGD).
/// </summary>
public sealed class FastLogisticRegression : IPredictor
{
    private readonly float _learningRate;
    private readonly int _epochs;
    private readonly float _l2Regularization;
    private float[] _weights = Array.Empty<float>();
    private float _bias;

    public float[] Weights => _weights;
    public float Bias => _bias;

    public FastLogisticRegression(float learningRate = 0.05f, int epochs = 50, float l2Regularization = 0.001f)
    {
        _learningRate = learningRate;
        _epochs = epochs;
        _l2Regularization = l2Regularization;
    }

    public void Fit(FeatureMatrix features, ReadOnlySpan<float> targets)
    {
        int rows = features.Rows;
        int cols = features.Columns;

        _weights = new float[cols];
        _bias = 0f;

        // SGD Optimization loop
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Predict(FeatureMatrix features, Span<float> predictions)
    {
        int rows = features.Rows;
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
