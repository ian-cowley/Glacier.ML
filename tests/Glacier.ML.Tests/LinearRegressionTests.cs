using System;
using Glacier.ML.Core;
using Glacier.ML.Linear;
using Xunit;

namespace Glacier.ML.Tests;

public class LinearRegressionTests
{
    [Fact]
    public void FastLinearRegression_OrdinaryLeastSquares_RecoversTrueCoefficients()
    {
        // Ground truth model: y = 3.0 * x0 - 2.0 * x1 + 1.5 * x2 + 5.0
        const int rows = 1000;
        const int cols = 3;

        using var matrix = new FeatureMatrix(rows, cols);
        float[] targets = new float[rows];

        var rng = new Random(42);
        for (int r = 0; r < rows; r++)
        {
            float x0 = (float)rng.NextDouble() * 10f;
            float x1 = (float)rng.NextDouble() * 10f;
            float x2 = (float)rng.NextDouble() * 10f;

            matrix.SetValue(r, 0, x0);
            matrix.SetValue(r, 1, x1);
            matrix.SetValue(r, 2, x2);

            targets[r] = 3.0f * x0 - 2.0f * x1 + 1.5f * x2 + 5.0f;
        }

        var reg = new FastLinearRegression(alpha: 0.0f, fitIntercept: true, target: GpuTarget.Auto);
        reg.Fit(matrix, targets);

        // Verify recovered weights and intercept
        Assert.Equal(3, reg.Weights.Length);
        Assert.True(MathF.Abs(reg.Weights[0] - 3.0f) < 0.05f, $"Expected weight[0] ~ 3.0, got {reg.Weights[0]}");
        Assert.True(MathF.Abs(reg.Weights[1] - (-2.0f)) < 0.05f, $"Expected weight[1] ~ -2.0, got {reg.Weights[1]}");
        Assert.True(MathF.Abs(reg.Weights[2] - 1.5f) < 0.05f, $"Expected weight[2] ~ 1.5, got {reg.Weights[2]}");
        Assert.True(MathF.Abs(reg.Intercept - 5.0f) < 0.05f, $"Expected intercept ~ 5.0, got {reg.Intercept}");

        // Predict
        float[] predictions = new float[rows];
        reg.Predict(matrix, predictions);

        float r2 = Metrics.R2Score(targets, predictions);
        Assert.True(r2 > 0.999f, $"Expected R2 > 0.999, got {r2}");
    }

    [Fact]
    public void FastLinearRegression_RidgeRegularization_ShrinksWeights()
    {
        const int rows = 500;
        const int cols = 2;

        using var matrix = new FeatureMatrix(rows, cols);
        float[] targets = new float[rows];

        var rng = new Random(42);
        for (int r = 0; r < rows; r++)
        {
            float x0 = (float)rng.NextDouble() * 5f;
            float x1 = (float)rng.NextDouble() * 5f;
            matrix.SetValue(r, 0, x0);
            matrix.SetValue(r, 1, x1);
            targets[r] = 2.0f * x0 + 1.0f * x1 + 3.0f;
        }

        var ols = new FastLinearRegression(alpha: 0.0f, fitIntercept: true, target: GpuTarget.Cpu);
        ols.Fit(matrix, targets);

        var ridge = new FastLinearRegression(alpha: 50.0f, fitIntercept: true, target: GpuTarget.Cpu);
        ridge.Fit(matrix, targets);

        // Ridge L2 penalty should shrink norm of weights compared to OLS
        float normOls = MathF.Sqrt(ols.Weights[0] * ols.Weights[0] + ols.Weights[1] * ols.Weights[1]);
        float normRidge = MathF.Sqrt(ridge.Weights[0] * ridge.Weights[0] + ridge.Weights[1] * ridge.Weights[1]);

        Assert.True(normRidge < normOls, $"Ridge norm ({normRidge}) should be less than OLS norm ({normOls})");
    }

    [Fact]
    public void FastLinearRegression_ZeroAlpha_ExactOlsOnTinyScaleFeatures_NoHiddenBias()
    {
        // Features scaled by 1e-4 so X^T X ~ 1e-6; old hidden 1e-6f regularization distorted weights by ~50%.
        const int rows = 200;
        const int cols = 2;
        using var matrix = new FeatureMatrix(rows, cols);
        float[] targets = new float[rows];

        var rng = new Random(123);
        for (int r = 0; r < rows; r++)
        {
            float x0 = (float)(rng.NextDouble() * 2e-4 - 1e-4);
            float x1 = (float)(rng.NextDouble() * 2e-4 - 1e-4);
            matrix.SetValue(r, 0, x0);
            matrix.SetValue(r, 1, x1);
            targets[r] = 3.0f * x0 - 2.0f * x1 + 1.5f;
        }

        var reg = new FastLinearRegression(alpha: 0.0f, fitIntercept: true, target: GpuTarget.Cpu);
        reg.Fit(matrix, targets);

        Assert.InRange(reg.Weights[0], 2.99f, 3.01f);
        Assert.InRange(reg.Weights[1], -2.01f, -1.99f);
        Assert.InRange(reg.Intercept, 1.499f, 1.501f);
    }

    [Fact]
    public void FastLinearRegression_CollinearColumns_SolvesViaJacobiPseudoinverseWithoutExplosion()
    {
        const int rows = 100;
        using var matrix = new FeatureMatrix(rows, 3);
        float[] targets = new float[rows];

        var rng = new Random(77);
        for (int r = 0; r < rows; r++)
        {
            float x0 = (float)rng.NextDouble() * 10f;
            matrix.SetValue(r, 0, x0);
            matrix.SetValue(r, 1, x0);        // Duplicate column
            matrix.SetValue(r, 2, 2.0f * x0); // Exact linear combination
            targets[r] = 6.0f * x0 + 2.0f;
        }

        var reg = new FastLinearRegression(alpha: 0.0f, fitIntercept: true, target: GpuTarget.Cpu);
        reg.Fit(matrix, targets);

        for (int c = 0; c < 3; c++)
        {
            Assert.True(float.IsFinite(reg.Weights[c]));
            Assert.True(MathF.Abs(reg.Weights[c]) < 10f, $"Collinear weight[{c}] exploded to {reg.Weights[c]}");
        }

        float[] preds = new float[rows];
        reg.Predict(matrix, preds);
        Assert.True(Metrics.R2Score(targets, preds) > 0.9999f);
    }

    [Fact]
    public void FastLinearRegression_ValidationAndColumnarZeroCopy_Works()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new FastLinearRegression(alpha: -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new FastLinearRegression(alpha: float.NaN));

        var unfitted = new FastLinearRegression();
        using var m = new FeatureMatrix(4, 2);
        Assert.Throws<InvalidOperationException>(() => unfitted.Predict(m, new float[4]));
        Assert.Throws<InvalidOperationException>(() => unfitted.PredictRow(new float[2]));

        float[] c0 = [1f, 2f, 3f, 4f];
        float[] c1 = [2f, 4f, 6f, 8f];
        float[] y = [5f, 10f, 15f, 20f];
        using var colMat = new FeatureMatrix(new ReadOnlyMemory<float>[] { c0, c1 });
        unfitted.Fit(colMat, y);

        var dataField = typeof(FeatureMatrix).GetField("_data", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.Null(dataField?.GetValue(colMat));

        float[] preds = new float[4];
        unfitted.Predict(colMat, preds);
        Assert.Null(dataField?.GetValue(colMat));
        for (int i = 0; i < 4; i++) Assert.Equal(y[i], preds[i], precision: 3);

        Assert.Throws<ArgumentException>(() => unfitted.PredictRow(new float[1]));
        Assert.Throws<ArgumentException>(() => unfitted.PredictRow(new float[3]));
        Assert.Throws<ArgumentException>(() => unfitted.Predict(new FeatureMatrix(4, 3), new float[4]));
        Assert.Throws<NotSupportedException>(() => new FastLinearRegression(target: GpuTarget.Nvidia).Fit(m, y));
    }
}
