using System;
using Glacier.ML.Compute;
using Glacier.ML.Core;
using Xunit;

namespace Glacier.ML.Tests;

public class GpuRingBufferMlTests
{
    [Fact]
    public void VectorAdd_CpuAndAuto_ProducesAccurateResults()
    {
        const int length = 256;
        float[] a = new float[length];
        float[] b = new float[length];
        float[] destCpu = new float[length];
        float[] destAuto = new float[length];

        for (int i = 0; i < length; i++)
        {
            a[i] = i * 1.5f;
            b[i] = i * 2.5f;
        }

        GpuMlAccelerator.VectorAdd(a, b, destCpu, GpuTarget.Cpu);
        GpuMlAccelerator.VectorAdd(a, b, destAuto, GpuTarget.Auto);

        for (int i = 0; i < length; i++)
        {
            float expected = a[i] + b[i];
            Assert.Equal(expected, destCpu[i], precision: 4);
            Assert.Equal(expected, destAuto[i], precision: 4);
        }
    }

    [Fact]
    public void VectorFma_CpuAndAuto_ProducesAccurateResults()
    {
        const int length = 256;
        float[] a = new float[length];
        float[] b = new float[length];
        float[] destCpu = new float[length];
        float[] destAuto = new float[length];
        const float scalar = 3.0f;

        for (int i = 0; i < length; i++)
        {
            a[i] = i * 2.0f;
            b[i] = 10.0f;
        }

        GpuMlAccelerator.VectorFma(a, b, destCpu, scalar, GpuTarget.Cpu);
        GpuMlAccelerator.VectorFma(a, b, destAuto, scalar, GpuTarget.Auto);

        for (int i = 0; i < length; i++)
        {
            float expected = a[i] * scalar + b[i];
            Assert.Equal(expected, destCpu[i], precision: 4);
            Assert.Equal(expected, destAuto[i], precision: 4);
        }
    }

    [Fact]
    public void VectorOperations_GpuNvidiaTarget_WhenAvailable()
    {
        if (!GpuMlAccelerator.IsNvidiaAvailable) return;

        const int length = 256;
        float[] a = new float[length];
        float[] b = new float[length];
        float[] destAdd = new float[length];
        float[] destFma = new float[length];
        const float scalar = -0.05f;

        for (int i = 0; i < length; i++)
        {
            a[i] = 10.0f + i;
            b[i] = 50.0f + i * 2.0f;
        }

        GpuMlAccelerator.VectorAdd(a, b, destAdd, GpuTarget.Nvidia);
        GpuMlAccelerator.VectorFma(a, b, destFma, scalar, GpuTarget.Nvidia);

        for (int i = 0; i < length; i++)
        {
            float expectedAdd = a[i] + b[i];
            float expectedFma = a[i] * scalar + b[i];

            Assert.Equal(expectedAdd, destAdd[i], precision: 3);
            Assert.Equal(expectedFma, destFma[i], precision: 3);
        }
    }
}
