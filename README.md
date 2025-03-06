This project provides an efficient GPU-accelerated implementation of quantum stabilizer tableau canonicalization using CUDA and Julia. The implementation leverages custom CUDA kernels to accelerate key operations, such as column swapping, phase computation, and matrix updates, providing significant speedups over CPU-based approaches. Benchmarking results show a 12x faster processing time.

This wil also be a part of QuantumCliffod Library

 Key Features
CUDA Accelerated: Utilizes CUDA to offload computations to the GPU, significantly improving performance for large stabilizer tableau operations.
Custom Kernels: Implements custom CUDA kernels for:
Linear copy operations
Column and phase swapping
Popcount and phase computation
Pivot search and update operations
Efficient Memory Management: Optimizes memory usage on the GPU for large-scale quantum computations.
Benchmarking Support: Includes a benchmarking function to assess performance gains compared to CPU implementations.

Usage
1. Run the Main Script:
To execute the main script and test GPU acceleration, run:

2. Example Usage:
using QuantumClifford

cpu_stab = QuantumClifford.random_stabilizer(5)
gpu_stab = cpu_to_gpu_stabilizer(cpu_stab)
canonicalize_gpu!(gpu_stab; phases=true)
cpu_stab_converted = gpu_to_cpu_stabilizer(gpu_stab)

println("Canonicalized Stabilizer:")
println(cpu_stab_converted)

🛠 Dependencies
CUDA.jl: For GPU acceleration.
QuantumClifford.jl: For quantum stabilizer operations.
BenchmarkTools.jl: For benchmarking performance.
LinearAlgebra.jl: For matrix operations.
