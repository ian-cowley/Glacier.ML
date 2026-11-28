namespace Glacier.ML.Core;

/// <summary>
/// Specifies the hardware execution target for Glacier.ML accelerated operations.
/// </summary>
public enum GpuTarget
{
    /// <summary>
    /// Automatically selects NVIDIA CUDA when supported for the operation and batch size, otherwise SIMD CPU.
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Targets bare-metal NVIDIA CUDA acceleration where supported (KMeans assignment and ring-buffer vector ops).
    /// CPU-only operations throw <see cref="System.NotSupportedException"/> if explicitly forced to Nvidia.
    /// </summary>
    Nvidia = 1,

    /// <summary>
    /// Reserved for AMD ROCm/HIP acceleration (currently unsupported in Glacier.ML; throws <see cref="System.NotSupportedException"/>).
    /// </summary>
    Amd = 2,

    /// <summary>
    /// Reserved for dual-GPU workload distribution (currently unsupported in Glacier.ML; throws <see cref="System.NotSupportedException"/>).
    /// </summary>
    DualGpu = 3,

    /// <summary>
    /// Forces CPU-only execution with AVX-512 / AVX2 SIMD vectorization.
    /// </summary>
    Cpu = 4
}
