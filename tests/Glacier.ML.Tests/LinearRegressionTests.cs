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
}
