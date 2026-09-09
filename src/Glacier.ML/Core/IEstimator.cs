using System;

namespace Glacier.ML.Core;

/// <summary>
/// Defines an estimator that can learn parameters from a training feature dataset.
/// </summary>
public interface IEstimator<TModel>
{
    TModel Fit(FeatureMatrix features, ReadOnlySpan<float> targets);
}

/// <summary>
/// Defines an estimator for unsupervised learning (e.g. clustering).
/// </summary>
public interface IUnsupervisedEstimator<TModel>
{
    TModel Fit(FeatureMatrix features);
}

/// <summary>
/// Defines a model capable of predicting output values for feature rows.
/// </summary>
public interface IPredictor
{
    void Predict(FeatureMatrix features, Span<float> predictions);
    float PredictRow(ReadOnlySpan<float> row);
}

/// <summary>
/// Defines a transformer that applies feature transformations (e.g. scaling, encoding) in-place.
/// </summary>
public interface ITransformer
{
    void Transform(FeatureMatrix features);
    void TransformRow(ReadOnlySpan<float> input, Span<float> output);
}
