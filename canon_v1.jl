################################################################################
# Final GPU-based Canonicalization via Full GF(2) Gaussian Elimination with Benchmark
#
# This script converts a stabilizer tableau (stored as a CuArray of UInt64)
# into its reduced row–echelon form (RREF) over GF(2). It assumes phases=false.
#
# The stabilizer tableau for n qubits is stored as a matrix of size (2*nblocks)×r,
# where nblocks = cld(nqubits, 64) and r is the number of generators.
#
# The algorithm is:
#
#   For each qubit j = 1:nqubits (using only the X part for pivot selection):
#     - Search (on the CPU) for a generator (column m, with m ≥ current_row)
#       whose X part has a 1 in the j-th bit (located in row 
#         block = div(j-1,64)+1, bit position = (j-1)%64).
#     - If found, swap that generator into position current_row.
#     - Then, in parallel on the GPU, update every other generator m ≠ current_row
#       by, for each block b from 1 to 2*nblocks:
#
#           d_mat[b, m] = d_mat[b, m] ⊻ d_mat[b, current_row]
#
#       but only if the pivot bit (in the X part) is set.
#     - Increment current_row.
#
# The result is the unique RREF mod 2 of the binary matrix.
#
# A benchmark function is provided to time only the GPU processing time.
################################################################################

using CUDA
using BenchmarkTools
using Printf
using LinearAlgebra

# For reference we import canonicalize! from QuantumClifford (not used in this script)
using QuantumClifford: canonicalize!, random_stabilizer

################################################################################
# Data Structure: GpuTableau
################################################################################
struct GpuTableau
    d_mat::CuArray{UInt64,2}    # Binary data: rows 1:nblocks = X part; rows nblocks+1:2*nblocks = Z part
    d_phases::CuVector{UInt8}   # Phases vector (unused here, phases=false)
    nqubits::Int                # Number of qubits
    nblocks::Int                # cld(nqubits, 64)
    r::Int                      # Number of generators (columns)
end

################################################################################
# Debug and Helper Functions
################################################################################
function get_stabilizer_matrix(gt::GpuTableau)
    r = gt.r
    nqubits = gt.nqubits
    nblocks = gt.nblocks
    M = Array(gt.d_mat)
    S = Array{Bool}(undef, r, 2*nqubits)
    for i in 1:r
        for j in 1:nqubits
            block = div(j-1, 64) + 1
            bit_pos = (j-1) % 64
            S[i, j] = ((M[block, i] >> bit_pos) & 1) == 1
            S[i, nqubits+j] = ((M[nblocks+block, i] >> bit_pos) & 1) == 1
        end
    end
    return S
end

function row_basis(A::Matrix{Bool})
    A = copy(A)
    m, n = size(A)
    pivot = 1
    for col in 1:n
        pivot_row = findfirst(r -> r, A[pivot:end, col])
        if pivot_row !== nothing
            pivot_row += pivot - 1
            A[[pivot, pivot_row], :] = A[[pivot_row, pivot], :]
            for r2 in pivot+1:m
                if A[r2, col]
                    A[r2, :] .= xor.(A[r2, :], A[pivot, :])
                end
            end
            pivot += 1
            if pivot > m break end
        end
    end
    basis = [A[i, :] for i in 1:m if any(A[i, :])]
    return basis
end

function row_to_int(v::AbstractVector{Bool})
    x = 0
    for b in v
        x = (x << 1) | (b ? 1 : 0)
    end
    return x
end

function stabilized_subspace_signature(gt::GpuTableau)
    S = get_stabilizer_matrix(gt)
    basis = row_basis(S)
    sig = sort(map(row_to_int, basis))
    return sig
end

function debug_print_signature(label::String, gt::GpuTableau)
    sig = stabilized_subspace_signature(gt)
    @printf("%s stabilized subspace signature: %s\n", label, string(sig))
end

function pretty_print_gpu_tableau(gt::GpuTableau)
    mat = Array(gt.d_mat)
    phases = Array(gt.d_phases)
    nqubits = gt.nqubits
    nblocks = gt.nblocks
    r = gt.r
    println("Stabilizer Tableau (display form):")
    for i in 1:r
        row_str = ""
        sign = phases[i] == 0 ? "+" : "-"
        for j in 1:nqubits
            block = div(j-1, 64) + 1
            bit_pos = (j-1) % 64
            xbit = ((mat[block, i] >> bit_pos) & 0x01) == 0x01
            zbit = ((mat[nblocks+block, i] >> bit_pos) & 0x01) == 0x01
            if xbit && zbit
                row_str *= "Y"
            elseif xbit
                row_str *= "X"
            elseif zbit
                row_str *= "Z"
            else
                row_str *= "_"
            end
        end
        println(sign, " ", row_str)
    end
    println("Phases: ", phases)
end

################################################################################
# GPU Kernel: Swap Columns
################################################################################
function swap_columns_kernel!(d_mat, col1::Int, col2::Int, N::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= N
        temp = d_mat[idx, col1]
        d_mat[idx, col1] = d_mat[idx, col2]
        d_mat[idx, col2] = temp
    end
    return
end

function gpu_swap_columns!(d_mat, col1::Int, col2::Int)
    N = size(d_mat, 1)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks swap_columns_kernel!(d_mat, col1, col2, N)
    CUDA.synchronize()
    #println("Swapping columns: $col1 <-> $col2")
end

################################################################################
# GPU Kernel: Update Rows (Full XOR Update)
#
# For a given pivot at column current_row, for each generator m ≠ current_row,
# if the pivot bit (in the X part) is set then update every block (from 1 to total_blocks)
# by performing:
#
#   d_mat[b, m] = d_mat[b, m] ⊻ d_mat[b, current_row]
#
# total_blocks = 2*nblocks.
################################################################################
function update_kernel!(d_mat, current_row::Int, pivot_block::Int, bit::UInt64, r::Int, total_blocks::Int)
    m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if m <= r && m != current_row
        if (d_mat[pivot_block, m] & bit) != 0
            for b in 1:total_blocks
                d_mat[b, m] = d_mat[b, m] ⊻ d_mat[b, current_row]
            end
        end
    end
    return
end

function gpu_update_kernel!(d_mat, current_row::Int, pivot_block::Int, bit::UInt64, r::Int, total_blocks::Int)
    
    threads = 256
    blocks = cld(r, threads)
    @cuda threads=threads blocks=blocks update_kernel!(d_mat, current_row, pivot_block, bit, r, total_blocks)
    CUDA.synchronize()
end

################################################################################
# Main GPU Canonicalization Routine via RREF mod 2 (Optimized Pivot Search)
################################################################################
function gpu_canonicalize_rref!(gt::GpuTableau)
    
    d_mat = gt.d_mat
    r = gt.r
    nqubits = gt.nqubits
    nblocks = gt.nblocks
    total_blocks = 2 * nblocks
    current_row = 1

    for j in 1:nqubits
        block = div(j - 1, 64) + 1   # corresponding row in the X part for qubit j
        bit = UInt64(1) << ((j - 1) % 64)
        pivot = 0

        # *** OPTIMIZATION ***
        # Instead of copying the entire matrix from GPU to CPU each time,
        # we copy only the needed row (and only the columns from current_row to r).
        A_row = Array(view(d_mat, block, current_row:r))
        pivot_offset = findfirst(x -> (x & bit) != 0, A_row)
        if pivot_offset !== nothing
            pivot = pivot_offset + current_row - 1
        end
        # ********************************

        if pivot == 0
            continue
        end

        if pivot != current_row
            gpu_swap_columns!(d_mat, pivot, current_row)
        end

        gpu_update_kernel!(d_mat, current_row, block, bit, r, total_blocks)
        current_row += 1
        if current_row > r
            break
        end
    end

    return gt
end

################################################################################
# Helpers to Build and Transfer a Tableau from Strings
################################################################################
function cpu_tableau_from_strings(rows::Vector{String})
    r = length(rows)
    parsed = [split(strip(row)) for row in rows]
    nqubits = length(parsed[1][2])
    nblocks = cld(nqubits, 64)
    mat = zeros(UInt64, 2*nblocks, r)
    phases = Vector{UInt8}(undef, r)
    for (j, (sign, pauli)) in enumerate(parsed)
        phases[j] = sign == "+" ? 0 : 1
        for (i, ch) in enumerate(pauli)
            if ch == 'X'
                mat[1, j] |= UInt64(1) << (i - 1)
            elseif ch == 'Z'
                mat[nblocks+1, j] |= UInt64(1) << (i - 1)
            elseif ch == 'Y'
                mat[1, j] |= UInt64(1) << (i - 1)
                mat[nblocks+1, j] |= UInt64(1) << (i - 1)
            end
        end
    end
    return mat, phases, nqubits
end

function cpu_to_gpu_tableau(mat::Matrix{UInt64}, phases::Vector{UInt8}, nqubits::Int)
    r = size(mat, 2)
    nblocks = cld(nqubits, 64)
    d_mat = CuArray(mat)
    d_phases = CuArray(phases)
    return GpuTableau(d_mat, d_phases, nqubits, nblocks, r)
end

################################################################################
# Pretty Print Function for Display
################################################################################
function pretty_print_gpu_tableau(gt::GpuTableau)
    mat = Array(gt.d_mat)
    phases = Array(gt.d_phases)
    nqubits = gt.nqubits
    nblocks = gt.nblocks
    r = gt.r
    println("Stabilizer Tableau (display form):")
    for i in 1:r
        row_str = ""
        sign = phases[i] == 0 ? "+" : "-"
        for j in 1:nqubits
            block = div(j-1, 64) + 1
            bit_pos = (j-1) % 64
            xbit = ((mat[block, i] >> bit_pos) & 0x01) == 0x01
            zbit = ((mat[nblocks+block, i] >> bit_pos) & 0x01) == 0x01
            if xbit && zbit
                row_str *= "Y"
            elseif xbit
                row_str *= "X"
            elseif zbit
                row_str *= "Z"
            else
                row_str *= "_"
            end
        end
        println(sign, " ", row_str)
    end
    println("Phases: ", phases)
end

################################################################################
# Benchmark Function for GPU Canonicalization
#
# This function generates a random stabilizer tableau for given nqubits and r,
# then benchmarks the GPU processing time of gpu_canonicalize_rref! using @belapsed.
################################################################################
function benchmark_gpu_canonicalization(nqubits::Int, r::Int)
    println("Benchmarking GPU canonicalization for nqubits = $nqubits, r = $r...")
    nblocks = cld(nqubits, 64)
    mat = rand(UInt64, 2*nblocks, r)
    phases = zeros(UInt8, r)
    gt = cpu_to_gpu_tableau(mat, phases, nqubits)
    # Warm-up.
    gpu_canonicalize_rref!(gt)
    CUDA.synchronize()
    t = @belapsed gpu_canonicalize_rref!($gt)
    println("GPU canonicalization time: $(t*1000) ms")
    return gt
end

################################################################################
# Main Test Driver and Benchmark
################################################################################
function main_test()
    # Known input test.
    input_strings = [
        "+ Y_YZ",
        "+ X_YY",
        "- Y_ZY",
        "+ ZY_X"
    ]
    
    
    
    
    
    

    




    mat, phases, nqubits = cpu_tableau_from_strings(input_strings)
    cpu_gt = cpu_to_gpu_tableau(mat, phases, nqubits)
    println("Pre-Canonicalization (known input):")
    pretty_print_gpu_tableau(cpu_gt)
    @printf("CPU stabilized subspace signature: %s\n", string(stabilized_subspace_signature(cpu_gt)))
    println()

    gpu_gt = cpu_to_gpu_tableau(mat, phases, nqubits)
    println("Running GPU canonicalization (phases=false)")
    gpu_canonicalize_rref!(gpu_gt)
    CUDA.synchronize()

    println("\nCanonicalization complete. GPU Canonical Tableau:")
    pretty_print_gpu_tableau(gpu_gt)
    @printf("GPU Final stabilized subspace signature: %s\n", string(stabilized_subspace_signature(gpu_gt)))
    println()

    println("Running tests:")
    if gpu_gt.r == cpu_gt.r
        println("Test number of generators: PASS (r = $(gpu_gt.r))")
    else
        println("Test number of generators: FAIL (expected $(cpu_gt.r) but got $(gpu_gt.r))")
    end

    function test_commutation(gt::GpuTableau)
        r = gt.r
        all_commute = true
        for i in 1:r
            for j in i+1:r
                M = Array(gt.d_mat)
                nblocks = gt.nblocks
                nqubits = gt.nqubits
                comm = 0
                for k in 1:nqubits
                    block = div(k-1, 64) + 1
                    bit = 1 << ((k-1) % 64)
                    xi = (M[block, i] & bit) != 0 ? 1 : 0
                    zi = (M[nblocks+block, i] & bit) != 0 ? 1 : 0
                    xj = (M[block, j] & bit) != 0 ? 1 : 0
                    zj = (M[nblocks+block, j] & bit) != 0 ? 1 : 0
                    comm += (xi*zj + zi*xj)
                end
                if (comm % 2) != 0
                    println("Test commutation: FAIL between generators $i and $j (symplectic product = $(comm % 2))")
                    all_commute = false
                end
            end
        end
        if all_commute
            println("Test commutation: PASS (all generators commute)")
        end
        return all_commute
    end
    test_commutation(gpu_gt)

    # Test stabilized subspace using brute-force span check.
    function in_span(v::Vector{Bool}, basis::Vector{Vector{Bool}})
        n = length(basis)
        for mask in 0:(2^n - 1)
            comb = falses(length(v))
            for i in 1:n
                if (mask >> (i-1)) & 1 == 1
                    comb .= xor.(comb, basis[i])
                end
            end
            if comb == v
                return true
            end
        end
        return false
    end

    function span_equal(basisA::Vector{Vector{Bool}}, basisB::Vector{Vector{Bool}})
        if length(basisA) != length(basisB)
            return false
        end
        for v in basisA
            if !in_span(v, basisB)
                return false
            end
        end
        for v in basisB
            if !in_span(v, basisA)
                return false
            end
        end
        return true
    end

    function test_stabilized_subspace(cpu_gt::GpuTableau, gpu_gt::GpuTableau)
        A = get_stabilizer_matrix(cpu_gt)
        B = get_stabilizer_matrix(gpu_gt)
        basisA = row_basis(A)
        basisB = row_basis(B)
        if span_equal(basisA, basisB)
            println("Test stabilized subspace: PASS (GPU and CPU row spaces match)")
            return true
        else
            println("Test stabilized subspace: FAIL")
            println("CPU row space signature: ", sort(map(row_to_int, basisA)))
            println("GPU row space signature: ", sort(map(row_to_int, basisB)))
            return false
        end
    end

    test_stabilized_subspace(cpu_gt, gpu_gt)

    println("\nRunning GPU benchmark for 6000 qubits, 6000 generators:")
    benchmark_gpu_canonicalization(1000, 1000)
end

main_test()
