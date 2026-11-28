using System;

namespace Glacier.ML.Core;

/// <summary>
/// Self-contained double-precision Symmetric Cyclic Jacobi eigensolver.
/// Computes all eigenvalues and orthonormal eigenvectors of a real symmetric matrix.
/// </summary>
internal static class SymmetricEigenSolver
{
    public static void SolveDescending(
        ReadOnlySpan<double> symmetricMatrix,
        int n,
        out double[] eigenvalues,
        out double[] eigenvectorsRowMajor)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (symmetricMatrix.Length < n * n)
            throw new ArgumentException("Matrix buffer smaller than n * n.", nameof(symmetricMatrix));

        double[] a = symmetricMatrix.Slice(0, n * n).ToArray();
        double[] v = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            v[i * n + i] = 1.0;
        }

        const int maxSweeps = 64;
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            double offDiagSq = 0.0;
            double diagSq = 0.0;
            for (int p = 0; p < n; p++)
            {
                double app = a[p * n + p];
                diagSq += app * app;
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p * n + q];
                    offDiagSq += apq * apq;
                }
            }

            if (offDiagSq <= 1e-28 * Math.Max(diagSq, 1e-300))
                break;

            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    double apq = a[p * n + q];
                    double app = a[p * n + p];
                    double aqq = a[q * n + q];
                    double scale = Math.Abs(app) + Math.Abs(aqq);

                    if (Math.Abs(apq) <= 1e-17 * Math.Max(scale, 1e-300))
                        continue;

                    double tau = (aqq - app) / (2.0 * apq);
                    double t = tau >= 0.0
                        ? 1.0 / (tau + Math.Sqrt(1.0 + tau * tau))
                        : -1.0 / (-tau + Math.Sqrt(1.0 + tau * tau));
                    double c = 1.0 / Math.Sqrt(1.0 + t * t);
                    double s = t * c;

                    a[p * n + p] = app - t * apq;
                    a[q * n + q] = aqq + t * apq;
                    a[p * n + q] = 0.0;
                    a[q * n + p] = 0.0;

                    for (int r = 0; r < n; r++)
                    {
                        if (r == p || r == q) continue;
                        double arp = a[r * n + p];
                        double arq = a[r * n + q];
                        double newArp = c * arp - s * arq;
                        double newArq = s * arp + c * arq;
                        a[r * n + p] = newArp;
                        a[p * n + r] = newArp;
                        a[r * n + q] = newArq;
                        a[q * n + r] = newArq;
                    }

                    int rowP = p * n;
                    int rowQ = q * n;
                    for (int r = 0; r < n; r++)
                    {
                        double vpr = v[rowP + r];
                        double vqr = v[rowQ + r];
                        v[rowP + r] = c * vpr - s * vqr;
                        v[rowQ + r] = s * vpr + c * vqr;
                    }
                }
            }
        }

        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (i, j) => a[j * n + j].CompareTo(a[i * n + i]));

        eigenvalues = new double[n];
        eigenvectorsRowMajor = new double[n * n];

        for (int i = 0; i < n; i++)
        {
            int src = order[i];
            eigenvalues[i] = a[src * n + src];

            int srcOffset = src * n;
            int dstOffset = i * n;

            int maxAbsIdx = 0;
            double maxAbs = 0.0;
            for (int d = 0; d < n; d++)
            {
                double val = v[srcOffset + d];
                double abs = Math.Abs(val);
                if (abs > maxAbs)
                {
                    maxAbs = abs;
                    maxAbsIdx = d;
                }
            }

            double sign = v[srcOffset + maxAbsIdx] < 0.0 ? -1.0 : 1.0;
            for (int d = 0; d < n; d++)
            {
                eigenvectorsRowMajor[dstOffset + d] = v[srcOffset + d] * sign;
            }
        }
    }
}
