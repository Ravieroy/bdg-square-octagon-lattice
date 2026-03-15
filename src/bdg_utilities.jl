module BdGUtilities
using DelimitedFiles
using LinearAlgebra
using Plots
using DataFrames
using CSV
using LaTeXStrings
using Statistics
using PyCall
@pyimport numpy as np

include("../src/general_utils.jl")
using .GeneralUtils

include("../src/logging_utils.jl")
using .LoggingUtils

# alias to fermi function
const f = fermi_fn

# module exports
export extract_geometry
export generate_t_map
export generate_t_matrix
export create_bdg_ham_up
export create_bdg_ham_dn
export build_H_twisted
export build_t_matrix
export create_bdg_ham_full
export create_bdg_ham_full_twisted
export sort_evecs
export sort_evecs_spinfull
export check_norm
export check_norm_spinful
export calc_delta
export calc_delta_spinfull
export calc_avg_n
export calc_avg_n_spinfull
export run_self_consistency_numpy
export run_self_consistency_numpy_spinfull
export run_self_consistency_numpy_spinfull_twisted
export bdg_spectrum_twist
export bdg_free_energy
export free_energy_twist
export compute_Ds_helicity
export compute_pairing_correlators
export triplet_diagnostics



"""
    extract_geometry(df; formatted=false)

Arguments:
- `df`: dataframe containing square-octagon lattice data.
- `formatted`: if `true`, expects columns `n1:n4`; if `false`, expects `r1:r4`.

Returns:
- `x, y, neighbors`: x/y coordinates and a neighbor list per site.
"""
function extract_geometry(df; formatted::Bool=false)
    nSites = nrow(df)
    x = Vector{Float64}(df.x)
    y = Vector{Float64}(df.y)

    if formatted
        @assert all(name -> hasproperty(df, name), (:n1, :n2, :n3, :n4)) "formatted=true requires columns n1,n2,n3,n4"
        neighbors = [Int[df.n1[i], df.n2[i], df.n3[i], df.n4[i]] for i in 1:nSites]
    else
        @assert all(name -> hasproperty(df, name), (:r1, :r2, :r3, :r4)) "formatted=false requires columns r1,r2,r3,r4"
        neighbors = [Int[df.r1[i], df.r2[i], df.r3[i], df.r4[i]] for i in 1:nSites]
    end

    return x, y, neighbors
end


"""
    generate_t_map(neighbors; t1=1.0, t2=1.0)

Build a symmetric hopping map `tMap[(i,j)] = t` from a neighbor list.

Arguments:
- `neighbors`: vector where `neighbors[i] == [r1,r2,r3,r4]` with r1–r3 = NN and r4 = NNN
- `t1`: NN hopping (default `1.0`)
- `t2`: NNN hopping for r4 (default `1.0`)

Returns:
- `Dict{Tuple{Int,Int},Float64}` mapping site pairs `(i,j)` to hopping values.
"""
function generate_t_map(neighbors::Vector{Vector{Int}}; t1::Float64=1.0, t2::Float64=1.0)
    nSites = length(neighbors)
    tMap = Dict{Tuple{Int,Int},Float64}()

    for i in 1:nSites
        @assert length(neighbors[i]) == 4 "neighbors[$i] must be [r1,r2,r3,r4] (r1–r3 NN, r4 NNN)."
        for (j_idx, j) in enumerate(neighbors[i])
            tVal = (j_idx == 4) ? t2 : t1
            tMap[(i, j)] = tVal
            tMap[(j, i)] = tVal
        end
    end

    return tMap
end


"""
    generate_t_matrix(fileName::String; save::Bool=false, outFile::String="tMat.csv")::Matrix{ComplexF64}

Reads a real-valued text file with interleaved real and imaginary parts of a Hamiltonian,
reconstructs the complex matrix, and optionally saves it. This file is generated from an
external Fortran program.

Arguments
- `fileName::String`: Path to the input file with rows of [Re, Im, Re, Im, ...].
- `save::Bool`: Save the reconstructed matrix to `outFile` (default: `false`).
- `outFile::String`: Output file name for saving the matrix (default: `"tMat.csv"`).

Returns
- `H::Matrix{ComplexF64}`: The reconstructed complex Hamiltonian matrix.
"""
function generate_t_matrix(fileName::String; save::Bool=false, outFile::String="tMat.csv")::Matrix{ComplexF64}
    rawHam = Matrix{Float64}(readdlm(fileName))

    nrows = size(rawHam, 1)
    ncols = size(rawHam, 1)

    H = zeros(ComplexF64, nrows, ncols)

    for row in 1:nrows
        c = 1
        for col in 1:ncols
            H[row, col] = rawHam[row, c] + im * rawHam[row, c+1]
            c += 2
        end
    end

    if save
        # Check if all imaginary parts are zero (within tolerance)
        is_real_matrix = all(abs.(imag.(H)) .< 1e-12)

        if is_real_matrix
            writedlm(outFile, real(H), ',')
            println("Saved H to $outFile as real matrix (imag parts ~0).")
        else
            H_str = string.(H)
            writedlm(outFile, H_str, ',')
            println("Saved H to $outFile as complex matrix (human-readable).")
        end
    end

    return H
end



"""
    create_bdg_ham_up(deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
                      K=0, isComplex=false, include_hartree=false)

2n x 2n BdG Hamiltonian for the decoupled sector in the basis (c_up; c_dn^dagger).

Optionally includes Hartree shifts (U*nDn on the up-particle block, U*nUp on the down-hole block).
"""
function create_bdg_ham_up(
    deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
    K=0, isComplex=false, includeHartree::Bool=false
)
    sigmaUp = 1
    sigmaDn = -1

    HBdG = isComplex ? zeros(ComplexF64, 2nSites, 2nSites) : zeros(Float64, 2nSites, 2nSites)

    # ----- particle block diagonal-----
    for i in 1:nSites
        hart = includeHartree ? (U * nDn[i]) : 0.0
        if i == impuritySite
            H[i, i] = -(mu + hart + (K - sigmaUp * J))
        else
            H[i, i] = -(mu + hart)
        end
    end

    # Fill particle/hole blocks from H
    @inbounds for i in 1:nSites, j in 1:nSites
        hij = H[i, j]
        HBdG[i, j] = hij
        HBdG[nSites+i, nSites+j] = -conj(hij)
    end

    # Pairing (onsite singlet)
    @inbounds for i in 1:nSites
        Δi = deltaList[i]
        HBdG[i, nSites+i] = Δi
        HBdG[nSites+i, i] = conj(Δi)
    end

    # ----- hole block diagonal -----
    # This overwrites the diagonal of the lower-right block to include mu, Hartree, impurities.
    @inbounds for i in 1:nSites
        hart = includeHartree ? (U * nUp[i]) : 0.0
        if i == impuritySite
            HBdG[nSites+i, nSites+i] = (mu + hart + (K - sigmaDn * J))
        else
            HBdG[nSites+i, nSites+i] = (mu + hart)
        end
    end

    return HBdG
end



"""
    create_bdg_ham_dn(deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
                      K=0, isComplex=false, include_hartree=false)

2n x 2n BdG Hamiltonian for the decoupled sector in the basis (c_dn; c_up^dagger).

Optionally includes Hartree shifts (U*nUp on the down-particle block, U*nDn on the up-hole block).
"""
function create_bdg_ham_dn(
    deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
    K=0, isComplex=false, includeHartree::Bool=false
)
    sigmaUp = 1
    sigmaDn = -1

    HBdG = isComplex ? zeros(ComplexF64, 2nSites, 2nSites) : zeros(Float64, 2nSites, 2nSites)

    # ----- particle block diagonal -----
    for i in 1:nSites
        hart = includeHartree ? (U * nUp[i]) : 0.0
        if i == impuritySite
            H[i, i] = -(mu + hart + (K - sigmaDn * J))
        else
            H[i, i] = -(mu + hart)
        end
    end

    # Fill particle/hole blocks from H
    @inbounds for i in 1:nSites, j in 1:nSites
        hij = H[i, j]
        HBdG[i, j] = hij
        HBdG[nSites+i, nSites+j] = -conj(hij)
    end

    # Pairing
    @inbounds for i in 1:nSites
        Δi = deltaList[i]
        HBdG[i, nSites+i] = Δi
        HBdG[nSites+i, i] = conj(Δi)
    end

    # ----- hole block diagonal -----
    @inbounds for i in 1:nSites
        hart = includeHartree ? (U * nDn[i]) : 0.0
        if i == impuritySite
            HBdG[nSites+i, nSites+i] = (mu + hart + (K - sigmaUp * J))
        else
            HBdG[nSites+i, nSites+i] = (mu + hart)
        end
    end

    return HBdG
end



function build_H_twisted(x::Vector{Float64},
    neighbors::Vector{Vector{Int}},
    tMap::Dict{Tuple{Int,Int},Float64},
    twist::Float64)

    nSites = length(x)
    H = zeros(ComplexF64, nSites, nSites)

    xmin, xmax = minimum(x), maximum(x)
    Lx = xmax - xmin

    for i in 1:nSites
        xi = x[i]
        for j in neighbors[i]
            tij = get(tMap, (i, j), 0.0)
            tij == 0.0 && continue

            dx = x[j] - xi
            dx -= round(dx / Lx) * Lx

            phase = exp(1im * twist * dx / Lx)
            hij = tij * phase

            H[i, j] = hij
            H[j, i] = conj(hij)
        end
    end

    return H, Lx
end



"""
    build_t_matrix(neighbors, tMap; onsite=0.0, isComplex=false)

Construct the n×n tight-binding matrix tMat from a neighbor list and hopping map.
Assumes `neighbors[i]` lists the neighbors of site i (e.g. [r1,r2,r3,r4]).

Arguments:
- `neighbors`: Vector{Vector{Int}} neighbor list
- `tMap`: Dict{Tuple{Int,Int},Float64} with hopping amplitudes
- `onsite`: onsite energy added to the diagonal (default 0.0)
- `isComplex`: if true, returns ComplexF64 matrix

Returns:
- `tMat`: nxn hopping matrix
"""
function build_t_matrix(
    neighbors::Vector{Vector{Int}},
    tMap::Dict{Tuple{Int,Int},Float64};
    onsite::Float64=0.0,
    isComplex::Bool=false
)
    nSites = length(neighbors)
    T = isComplex ? ComplexF64 : Float64
    tMat = zeros(T, nSites, nSites)

    # onsite term (if any)
    if onsite != 0.0
        @inbounds for i in 1:nSites
            tMat[i,i] = onsite
        end
    end

    # fill hoppings
    @inbounds for i in 1:nSites
        for j in neighbors[i]
            tij = get(tMap, (i,j), 0.0)
            tij == 0.0 && continue
            tMat[i,j] += tij
        end
    end

    return tMat
end



"""
Build the full spinful 4n x 4n BdG Hamiltonian in the basis

(c1_up ... cn_up; c1_dn^dagger ... cn_dn^dagger; c1_dn ... cn_dn; c1_up^dagger ... cn_up^dagger).

This function first constructs the spinful normal-state Hamiltonian, including
Rashba SOC and, if supplied, intrinsic SOC. It then adds onsite singlet pairing,
forms the standard BdG matrix, and finally reorders it into the basis above.

Arguments
- `deltaList`: onsite superconducting gap values for each site
- `H`: spinless n x n tight-binding matrix
- `mu`: chemical potential
- `nSites`: number of lattice sites
- `nUp`, `nDn`: spin-up and spin-down densities, used only if Hartree terms are included
- `U`: Hubbard interaction strength, used only if Hartree terms are included
- `impuritySite`: impurity site index (1-based, or `0` if there is no impurity)
- `J`: magnetic impurity strength
- `x`, `y`: site coordinates
- `neighbors`: bond list used to build the SOC terms
- `lambda`: Rashba SOC strength
- `tMap`: dictionary of bond-dependent hoppings `(i,j) => t_ij`

Keyword arguments
- `K=0.0`: non-magnetic impurity strength
- `include_hartree=false`: if `true`, adds Hartree shifts `-U*n_{-sigma}` to the particle sector
- `isoMap=nothing`: optional intrinsic SOC sign map `(i,j) => nu_ij`
- `lambdaIso=0.0`: intrinsic SOC strength

Returns
- `Hfull`: the full 4n x 4n BdG Hamiltonian
"""
function create_bdg_ham_full(deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J,
    x, y, neighbors, lambda, tMap;
    K=0.0, includeHartree::Bool=false,
    isoMap=nothing, lambdaIso=0.0)

    n = nSites
    @assert length(deltaList) == n
    @assert size(H, 1) == n && size(H, 2) == n

    # choose element type
    T = promote_type(eltype(H), eltype(deltaList), ComplexF64)

    # Spinful normal h
    h = zeros(T, 2n, 2n)

    h[1:n, 1:n] .= T.(H)
    h[n+1:2n, n+1:2n] .= T.(H)

    # add chemical potential + impurities (+ optional Hartree)
    for i in 1:n
        # impurity shifts
        v_up = -mu
        v_dn = -mu

        if impuritySite != 0 && i == impuritySite
            v_up = -(mu + (K - (+1) * J))
            v_dn = -(mu + (K - (-1) * J))
        end

        if includeHartree
            v_up -= U * nDn[i]
            v_dn -= U * nUp[i]
        end

        h[i, i] += v_up
        h[n+i, n+i] += v_dn
    end

    # Rashba SOC
    if lambda != 0.0
        for i in 1:n
            xi, yi = x[i], y[i]
            for j in neighbors[i]
                tij = get(tMap, (i, j), 0.0)
                tij == 0.0 && continue

                dx = x[j] - xi
                dy = y[j] - yi

                S = (-im) * lambda * tij * (-dy - im * dx)

                h[i, n+j] += S
                h[n+j, i] += conj(S)
            end
        end
    end

    # Optional intrinsic SOC
    if isoMap !== nothing && lambdaIso != 0.0
        for (key, ν) in isoMap
            i, j = key
            term = im * lambdaIso * ν
            h[i, j] += term
            h[n+i, n+j] -= term
            h[j, i] += conj(term)
            h[n+j, n+i] -= conj(term)
        end
    end

    Δ = zeros(T, 2n, 2n)
    for i in 1:n
        Δi = T(deltaList[i])
        Δ[i, n+i] = Δi
        Δ[n+i, i] = -Δi
    end

    # Standard BdG
    HB = zeros(T, 4n, 4n)
    HB[1:2n, 1:2n] .= h
    HB[1:2n, 2n+1:4n] .= Δ
    HB[2n+1:4n, 1:2n] .= adjoint(Δ)
    HB[2n+1:4n, 2n+1:4n] .= -transpose(h)

    # present basis
    perm = vcat(1:n, 3n+1:4n, n+1:2n, 2n+1:3n)
    Hfull = HB[perm, perm]

    return Hfull
end


"""
Build the full twisted 4n x 4n BdG Hamiltonian in the basis

(c1_up ... cn_up; c1_dn^dagger ... cn_dn^dagger; c1_dn ... cn_dn; c1_up^dagger ... cn_up^dagger).

This is the twisted version of the full spinful BdG Hamiltonian. A Peierls phase
is applied to all bond terms, including both the hopping and Rashba SOC terms.

Arguments
- `deltaList`: onsite gap values
- `mu`: chemical potential
- `nSites`: number of lattice sites
- `nUp`, `nDn`: spin densities, used only with Hartree terms
- `U`: Hubbard interaction, used only with Hartree terms
- `impuritySite`: impurity site index (1-based, or `0` for none)
- `J`: magnetic impurity strength
- `x`, `y`: site coordinates
- `neighbors`: bond list
- `lambda`: Rashba SOC strength
- `tMap`: bond-dependent hopping dictionary `(i,j) => t_ij`

Keywords
- `twist=0.0`: twist angle
- `Lx=nothing`: system length along the twist direction
- `K=0.0`: non-magnetic impurity strength
- `include_hartree=false`: include Hartree shifts
- `isoMap=nothing`: intrinsic SOC sign map
- `lambdaIso=0.0`: intrinsic SOC strength

Returns
- `Hfull`: full twisted 4n x 4n BdG Hamiltonian
"""
function create_bdg_ham_full_twisted(deltaList, mu, nSites, nUp, nDn, U, impuritySite, J,
    x, y, neighbors, lambda, tMap;
    twist=0.0, Lx=nothing, K=0.0,
    includeHartree::Bool=false,
    isoMap=nothing, lambdaIso=0.0)

    n = nSites
    T = promote_type(eltype(deltaList), ComplexF64)

    Hhop, Lx_val = build_H_twisted(x, neighbors, tMap, twist)

    h = zeros(T, 2n, 2n)
    h[1:n, 1:n] .= T.(Hhop)
    h[n+1:2n, n+1:2n] .= T.(Hhop)

    for i in 1:n
        v_up = -mu
        v_dn = -mu

        if impuritySite != 0 && i == impuritySite
            v_up = -(mu + (K - (+1) * J))
            v_dn = -(mu + (K - (-1) * J))
        end

        if includeHartree
            v_up -= U * nDn[i]
            v_dn -= U * nUp[i]
        end

        h[i, i] += v_up
        h[n+i, n+i] += v_dn
    end

    # Rashba SOC WITH the same twist phase on the bond i->j
    if lambda != 0.0
        for i in 1:n
            xi, yi = x[i], y[i]
            for j in neighbors[i]
                tij = get(tMap, (i, j), 0.0)
                tij == 0.0 && continue

                dx = x[j] - xi
                dy = y[j] - yi
                dx -= round(dx / Lx_val) * Lx_val

                phase = exp(1im * twist * dx / Lx_val)

                S0 = (-im) * lambda * tij * (-dy - im * dx)
                S = S0 * phase

                h[i, n+j] += S
                h[n+j, i] += conj(S)
            end
        end
    end

    if isoMap !== nothing && lambdaIso != 0.0
        for (key, ν) in isoMap
            i, j = key
            dx = x[j] - x[i]
            dx -= round(dx / Lx_val) * Lx_val
            phase = exp(1im * twist * dx / Lx_val)

            term = (im * lambdaIso * ν) * phase
            h[i, j] += term
            h[j, i] += conj(term)
            h[n+i, n+j] -= term
            h[n+j, n+i] -= conj(term)
        end
    end

    # singlet onsite delta
    Δ = zeros(T, 2n, 2n)
    for i in 1:n
        Δi = T(deltaList[i])
        Δ[i, n+i] = Δi
        Δ[n+i, i] = -Δi
    end

    HB = zeros(T, 4n, 4n)
    HB[1:2n, 1:2n] .= h
    HB[1:2n, 2n+1:4n] .= Δ
    HB[2n+1:4n, 1:2n] .= adjoint(Δ)
    HB[2n+1:4n, 2n+1:4n] .= -transpose(h)

    perm = vcat(1:n, 3n+1:4n, n+1:2n, 2n+1:3n)
    return HB[perm, perm]
end



"""
    sort_evecs(evector, nSites)

From a 2n x 2n BdG eigenvector matrix, whose columns are eigenvectors,
extract the positive-energy u and v blocks in shape (nSites x nSites).

Returns:
- `un`, `vn`: matrices of size (nSites x nSites), where `un[n,i]` and `vn[n,i]`
  correspond to the positive-energy state `n`
"""
function sort_evecs(evector::AbstractMatrix{<:Number}, nSites::Int)
    # numpy eigh returns eigenvectors in columns: evecs[:,k] is k-th eigenvector.
    # We want "state index" as first dimension: (nStates × 2nSites).
    evecsT = transpose(evector)

    # For a standard BdG ordering, positive energies are the last nSites states
    epos = evecsT[nSites+1:2*nSites, :]  # (nSites × 2nSites)

    un = epos[:, 1:nSites]
    vn = epos[:, nSites+1:2*nSites]
    return un, vn
end


"""
    sort_evecs_spinfull(evecs, evals, nSites; tol=0.0)

Select the positive-energy BdG eigenvectors and separate them into
(u_up, v_dn, u_dn, v_up) for the basis (c_up; c_dn^dagger; c_dn; c_up^dagger).

The output is arranged so that `u_up[n,i]` and the other components can be used
directly in `calc_delta_spinfull`.
"""
function sort_evecs_spinfull(
    evecs::AbstractMatrix{<:Number},
    evals::AbstractVector{<:Real},
    nSites::Int; tol::Real=0.0
)
    @assert size(evecs, 1) == 4nSites && size(evecs, 2) == 4nSites
    @assert length(evals) == 4nSites

    idx = findall(>(tol), evals)          # positive-energy indices
    VposT = transpose(evecs[:, idx])      # (nPos)×(4n) with no conjugation

    u_up = VposT[:, 1:nSites]
    v_dn = VposT[:, nSites+1:2nSites]
    u_dn = VposT[:, 2nSites+1:3nSites]
    v_up = VposT[:, 3nSites+1:4nSites]

    return u_up, v_dn, u_dn, v_up, evals[idx]
end


"""
    check_norm(un, vn, nSites; digits=5)

Check the normalization |un|^2 + |vn|^2 = 1 for each eigenstate.

Arguments:
- `un`, `vn`: particle and hole components
- `nSites`: number of eigenstates
- `digits`: rounding precision

Returns:
- vector of normalization values
"""
function check_norm(
    un::AbstractMatrix{<:Number},
    vn::AbstractMatrix{<:Number},
    nSites::Int; digits::Int=5
)
    normList = zeros(Float64, nSites)
    for n in 1:nSites
        normList[n] =
            sum(abs2, un[n, :]) +
            sum(abs2, vn[n, :])
    end
    return round.(normList, digits=digits)
end


"""
    check_norm_spinful(u_up, v_dn, u_dn, v_up, nSites; digits=5)

Check the normalization sum(|u|^2 + |v|^2) over spin components
for each eigenstate.

Arguments:
- `u_up`, `v_dn`, `u_dn`, `v_up`: BdG components
- `nSites`: number of eigenstates
- `digits`: rounding precision

Returns:
- vector of normalization values
"""
function check_norm_spinful(
    u_up::AbstractMatrix{<:Number},
    v_dn::AbstractMatrix{<:Number},
    u_dn::AbstractMatrix{<:Number},
    v_up::AbstractMatrix{<:Number},
    nSites::Int; digits::Int=5
)
    normList = zeros(Float64, nSites)
    for n in 1:nSites
        normList[n] =
            sum(abs2, u_up[n, :]) +
            sum(abs2, v_dn[n, :]) +
            sum(abs2, u_dn[n, :]) +
            sum(abs2, v_up[n, :])
    end
    return round.(normList, digits=digits)
end


"""
    calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, E1, E2, T)

Onsite singlet gap update for the non-spinful split solve.
Inputs are positive-energy amplitudes only (nSites states).

Returns:
- DeltaList : Vector{Float64} if purely real, else Vector{ComplexF64}
"""
function calc_delta(
    U,
    un_up::AbstractMatrix{<:Number},
    vn_up::AbstractMatrix{<:Number},
    un_dn::AbstractMatrix{<:Number},
    vn_dn::AbstractMatrix{<:Number},
    nSites::Int,
    E1::AbstractVector{<:Number},
    E2::AbstractVector{<:Number},
    T
)
    DeltaList = zeros(ComplexF64, nSites)

    @inbounds for i in 1:nSites
        Δ = 0.0 + 0.0im
        for n in 1:nSites
            f1 = f(E1[n]; T=T)
            f2 = f(E2[n]; T=T)

            Δ += U * (
                un_up[n, i] * conj(vn_dn[n, i]) * (1 - f1)
                -
                un_dn[n, i] * conj(vn_up[n, i]) * f2
            )
        end
        DeltaList[i] = Δ
    end

    return all(isreal, DeltaList) ? Float64.(DeltaList) : DeltaList
end


function calc_delta_spinfull(
    U,
    un_up::AbstractMatrix{<:Number},
    vn_up::AbstractMatrix{<:Number},
    un_dn::AbstractMatrix{<:Number},
    vn_dn::AbstractMatrix{<:Number},
    nSites::Int,
    Epos::AbstractVector{<:Real},
    T
)
    DeltaList = zeros(ComplexF64, nSites)
    nStates = length(Epos)
    @assert size(un_up, 1) == nStates

    for i in 1:nSites
        Δ = zero(ComplexF64)
        for n in 1:nStates
            fn = f(Epos[n]; T=T)
            Δ += U * (
                un_up[n, i] * conj(vn_dn[n, i]) * (1 - fn)
                -
                un_dn[n, i] * conj(vn_up[n, i]) * fn
            )
        end
        DeltaList[i] = Δ
    end

    return all(isreal, DeltaList) ? Float64.(DeltaList) : DeltaList
end


"""
    calc_avg_n(un, vn, nSites, E1, E2, T)

Site-resolved density update for one spin sector in the split (non-spinful) solve.
Uses positive-energy states only (nSites states).

Returns:
- avgNList :: Vector{Float64}
"""
function calc_avg_n(un, vn, nSites, E1, E2, T)
    avgNList = zeros(Float64, nSites)

    @inbounds for i in 1:nSites
        ni = 0.0
        for n in 1:nSites
            f1 = f(E1[n]; T=T)
            f2 = f(E2[n]; T=T)

            ni += abs2(un[n, i]) * f1 + abs2(vn[n, i]) * (1 - f2)
        end
        avgNList[i] = ni
    end

    return avgNList
end


function calc_avg_n_spinfull(un, vn, nSites, E, T)
    avgNList = zeros(Float64, nSites)
    nStates = size(un, 1)
    @assert size(vn, 1) == nStates
    @assert length(E) == nStates

    for i in 1:nSites
        avgN = 0.0
        for n in 1:nStates
            fn = f(E[n]; T=T)
            avgN += abs2(un[n, i]) * fn +
                    abs2(vn[n, i]) * (1 - fn)
        end
        avgNList[i] = avgN
    end

    return avgNList
end



"""
run_self_consistency_numpy( deltaOld, mu, nSites, nUp, nDn, tMat, U, J,
    impuritySite, T; tol=1e-5, maxCount=150, includeHartree=false,
    verboseLogIn=nothing
)

Self-consistent non-spinful BdG using two decoupled 2n×2n sectors built from
`create_bdg_ham_up` and `create_bdg_ham_dn` and solved via NumPy `eigh`.

Returns:
(deltaFinal, nUpFinal, nDnFinal, nAvg, evecs_up, evecs_dn, evals_up, evals_dn,
isConverged, endTime, count)
"""
function run_self_consistency_numpy(
    deltaOld, mu, nSites, nUp, nDn, tMat, U, J, impuritySite, T;
    tol=1e-5, maxCount=150, includeHartree=false, verboseLogIn=nothing
)
    startTime = time()
    count = 0
    flag = false

    deltaOld = copy(deltaOld)
    nUp = copy(nUp)
    nDn = copy(nDn)

    while flag == false
        count += 1
        innerLoopTime = time()

        H1 = create_bdg_ham_up(
            deltaOld, copy(tMat), mu, nSites, nUp, nDn, U, impuritySite, J,
            includeHartree=includeHartree)

        H2 = create_bdg_ham_dn(
            deltaOld, copy(tMat), mu, nSites, nUp, nDn, U, impuritySite, J,
            includeHartree=includeHartree)

        evals1, evecs1 = np.linalg.eigh(H1)
        evals2, evecs2 = np.linalg.eigh(H2)

        un_up, vn_dn = sort_evecs(evecs1, nSites)
        un_dn, vn_up = sort_evecs(evecs2, nSites)

        evalPositive1 = evals1[nSites+1:end]
        evalPositive2 = evals2[nSites+1:end]

        deltaNew = calc_delta(
            U, un_up, vn_up, un_dn, vn_dn,
            nSites, evalPositive1, evalPositive2, T
        )

        nUpNew = calc_avg_n(un_up, vn_dn, nSites, evalPositive1, evalPositive2, T)
        nDnNew = calc_avg_n(un_dn, vn_up, nSites, evalPositive1, evalPositive2, T)

        if check_rel_tol(deltaOld, deltaNew; tol=tol)
            deltaFinal = deltaNew
            nUpFinal = nUpNew
            nDnFinal = nDnNew
            nAvg = nUpFinal + nDnFinal
            endTime = round(time() - startTime, digits=2)
            return deltaFinal, nUpFinal, nDnFinal, nAvg,
            evecs1, evecs2, evals1, evals2,
            true, endTime, count

        else

            deltaOld = deltaNew
            nUp = nUpNew
            nDn = nDnNew
            if verboseLogIn !== nothing
                endTime = round(time() - innerLoopTime, digits=2)
                message = string("Iteration $count completed in ", format_elapsed_time(endTime))
                write_log(verboseLogIn, message) 
            end

            if count >= maxCount
                deltaFinal = deltaNew
                nUpFinal = nUpNew
                nDnFinal = nDnNew
                nAvg = nUpFinal + nDnFinal
                endTime = round(time() - startTime, digits=2)
                isConverged = false
                flag = true

                return deltaFinal, nUpFinal, nDnFinal, nAvg,
                evecs1, evecs2, evals1, evals2,
                isConverged, endTime, count
            end
        end
    end
end



function run_self_consistency_numpy_spinfull(
    deltaOld, mu, nSites, nUp, nDn, tMat, U, J,
    x, y, neighbors, lambda, lambdaIso, tMap, isoMap,
    impuritySite, T;
    tol=0.001, maxCount=100, K=0.0, includeHartree=false, verboseLogIn=nothing
)
    startTime = time()
    count = 0

    while true
        count += 1
        innerLoopTime = time()

        H_BdG = create_bdg_ham_full(
            deltaOld, tMat, mu, nSites, nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap;
            K=K, includeHartree=includeHartree, isoMap=isoMap,
            lambdaIso=lambdaIso
        )

        (evals, evecs) = np.linalg.eigh(H_BdG)

        un_up, vn_dn, un_dn, vn_up, Epos = sort_evecs_spinfull(evecs, evals, nSites; tol=0.0)

        deltaNew = calc_delta_spinfull(U, un_up, vn_up, un_dn, vn_dn, nSites, Epos, T)

        nUpNew = calc_avg_n_spinfull(un_up, vn_up, nSites, Epos, T)
        nDnNew = calc_avg_n_spinfull(un_dn, vn_dn, nSites, Epos, T)

        if check_rel_tol(deltaOld, deltaNew; tol=tol)
            deltaFinal = deltaNew
            nUpFinal = nUpNew
            nDnFinal = nDnNew
            nAvg = nUpFinal + nDnFinal
            isConverged = true
            endTime = round(time() - startTime, digits=2)

            return deltaFinal, nUpFinal, nDnFinal, nAvg,
            evecs, evals, isConverged, endTime, count
        end

        deltaOld = deltaNew
        nUp = nUpNew
        nDn = nDnNew
        if verboseLogIn != nothing
            endTime = round(time() - innerLoopTime, digits=2)
            message = string("Iteration $count completed in ", format_elapsed_time(endTime))
            write_log(verboseLogIn, message)
        end

        if count >= maxCount
            deltaFinal = deltaNew
            nUpFinal = nUpNew
            nDnFinal = nDnNew
            nAvg = nUpFinal + nDnFinal
            isConverged = false
            endTime = round(time() - startTime, digits=2)

            return deltaFinal, nUpFinal, nDnFinal, nAvg,
            evecs, evals, isConverged, endTime, count
        end
    end
end


"""
    run_self_consistency_numpy_spinfull_twisted(
        deltaOld, mu, nSites, nUp, nDn, U, J, x, y,
        neighbors, lambda, lambdaIso, tMap, isoMap, impuritySite, T;
        twist=0.0, K=0.0, tol=1e-3, maxCount=200,
        includeHartree=false, verboseLogIn=nothing
    )

Run the self-consistency loop for the spinful BdG Hamiltonian with optional
twist and spin-orbit coupling.

This routine uses the 4n x 4n BdG Hamiltonian in the basis
(c_up; c_dn^dagger; c_dn; c_up^dagger), selects the positive-energy states
using `evals .> 0`, and updates the gap and spin densities.

Arguments
- `deltaOld`: initial gap values at each site (length `nSites`, real or complex)
- `mu`: chemical potential
- `nSites`: number of lattice sites
- `nUp`, `nDn`: initial spin-up and spin-down densities
- `U`: attractive interaction strength
- `J`: magnetic impurity strength
- `x`, `y`: site coordinates
- `neighbors`: neighbor list
- `lambda`: Rashba SOC strength
- `lambdaIso`: intrinsic SOC strength
- `tMap`: hopping map used for SOC weighting and twisted hopping
- `isoMap`: intrinsic SOC map, or `nothing`
- `impuritySite`: impurity site index, or `0` for none
- `T`: temperature

Keyword arguments
- `twist=0.0`: phase twist
- `K=0.0`: non-magnetic impurity strength
- `tol=1e-3`: relative convergence tolerance for the gap
- `maxCount=200`: maximum number of iterations
- `includeHartree=false`: include Hartree shifts in the Hamiltonian
- `verboseLogIn=nothing`: optional logger handle

Returns
- `deltaFinal`: final gap values
- `nUpFinal`, `nDnFinal`: final spin densities
- `nAvg`: average density
- `evecs`, `evals`: eigenvectors and eigenvalues from the final step
- `isConverged`: whether the loop converged
- `endTime`: total runtime
- `count`: number of iterations performed
"""
function run_self_consistency_numpy_spinfull_twisted(
    deltaOld, mu, nSites, nUp, nDn, U, J, x, y,
    neighbors, lambda, lambdaIso, tMap, isoMap, impuritySite, T;
    twist=0.0, K=0.0, tol=1e-3, maxCount=200,
    includeHartree::Bool=false, verboseLogIn=nothing
)
    startTime = time()
    count = 0

    while true
        count += 1
        innerLoopTime = time()

        H_BdG = create_bdg_ham_full_twisted(
            deltaOld, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap;
            twist=twist, K=K, includeHartree=includeHartree,
            isoMap=isoMap, lambdaIso=lambdaIso
        )

        (evals, evecs) = np.linalg.eigh(H_BdG)

        # positive-energy selection + block slicing
        un_up, vn_dn, un_dn, vn_up, Epos = sort_evecs_spinfull(evecs, evals, nSites; tol=0.0)

        # gap update
        deltaNew = calc_delta_spinfull(U, un_up, vn_up, un_dn, vn_dn, nSites, Epos, T)

        # densities
        nUpNew = calc_avg_n_spinfull(un_up, vn_up, nSites, Epos, T)
        nDnNew = calc_avg_n_spinfull(un_dn, vn_dn, nSites, Epos, T)

        if check_rel_tol(deltaOld, deltaNew; tol=tol)
            deltaFinal = deltaNew
            nUpFinal = nUpNew
            nDnFinal = nDnNew
            nAvg = nUpFinal + nDnFinal
            isConverged = true
            endTime = round(time() - startTime, digits=2)

            return deltaFinal, nUpFinal, nDnFinal, nAvg,
            evecs, evals, isConverged, endTime, count
        end

        deltaOld = deltaNew
        nUp = nUpNew
        nDn = nDnNew

        if verboseLogIn !== nothing
            elapsed = round(time() - innerLoopTime, digits=2)
            message = string("Iteration $count completed in ", format_elapsed_time(elapsed))
            write_log(verboseLogIn, message)
        end

        if count >= maxCount
            deltaFinal = deltaNew
            nUpFinal = nUpNew
            nDnFinal = nDnNew
            nAvg = nUpFinal + nDnFinal
            isConverged = false
            endTime = round(time() - startTime, digits=2)

            return deltaFinal, nUpFinal, nDnFinal, nAvg,
            evecs, evals, isConverged, endTime, count
        end
    end
end



function bdg_spectrum_twist(
    deltaList, mu, nSites,
    nUp, nDn, U, impuritySite, J,
    x, y, neighbors, lambda, tMap;
    twist=0.0, K=0.0, Lx=nothing, isoMap=nothing, lambdaIso=0.0
)
    Hfull = create_bdg_ham_full_twisted(
        deltaList, mu, nSites,
        nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap;
        twist=twist, K=K, Lx=Lx, isoMap=isoMap, lambdaIso=lambdaIso
    )

    evals, _ = np.linalg.eigh(Hfull)
    return evals
end


"""
    bdg_free_energy(evals, deltaList, U, T)

Compute the BdG free energy.

Arguments:
- evals: Complete BdG eigenvalues.
- deltaList: Converged order parameter at all sites.
- U: Attractive interaction strength.
- T: Temperature.

Returns:
- Free energy (ground-state energy if T == 0, finite-temperature free energy otherwise).
"""
function bdg_free_energy(evals, deltaList, U, T)
    if T == 0.0
        # T = 0 ground-state energy
        return -0.5 * sum(abs.(evals)) + sum(abs2.(deltaList)) / U
    else
        # finite-T free energy
        return -T * sum(log.(2 .* cosh.(abs.(evals) ./ (2T)))) +
               sum(abs2.(deltaList)) / U
    end
end


function free_energy_twist(
    twist, deltaList, mu, nSites,
    nUp, nDn, U, impuritySite, J,
    x, y, neighbors, lambda, tMap, T;
    K=0.0, Lx=nothing, isoMap=nothing, lambdaIso=0.0
)
    evals = bdg_spectrum_twist(
        deltaList, mu, nSites,
        nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap;
        twist=twist, K=K, Lx=Lx, isoMap=isoMap, lambdaIso=lambdaIso
    )
    return bdg_free_energy(evals, deltaList, U, T)
end


function compute_Ds_helicity(
    deltaList, mu, nSites,
    nUp, nDn, U, impuritySite, J,
    x, y, neighbors, lambda, tMap, T;
    K=0.0, twist=1e-3, Lx=nothing, isoMap=nothing, lambdaIso=0.0
)
    xmin, xmax = minimum(x), maximum(x)
    Lx_val = isnothing(Lx) ? (xmax - xmin) : Lx

    F0 = free_energy_twist(0.0, deltaList, mu, nSites, nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap, T;
        K=K, Lx=Lx_val, isoMap=isoMap, lambdaIso=lambdaIso)

    Fp = free_energy_twist(+twist, deltaList, mu, nSites, nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap, T;
        K=K, Lx=Lx_val, isoMap=isoMap, lambdaIso=lambdaIso)

    Fm = free_energy_twist(-twist, deltaList, mu, nSites, nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap, T;
        K=K, Lx=Lx_val, isoMap=isoMap, lambdaIso=lambdaIso)

    Ds = (Lx_val^2 / nSites) * (Fp + Fm - 2F0) / twist^2
    return real(Ds)
end



"""
compute_pairing_correlators(un_up, vn_up, un_dn, vn_dn, E, T, nSites; tol=1e-12)

Compute the real-space anomalous correlators F_{ij}^{sigma sigma'} from a
spinful BdG solution using only the positive-energy states, and report
Pauli-symmetry diagnostics.

Inputs:
- `un_up`, `vn_up`: BdG amplitudes for spin up  (nStates x nSites)
- `un_dn`, `vn_dn`: BdG amplitudes for spin down  (nStates x nSites)
- `E`: positive BdG eigenvalues (length `nStates`)
- `T`: temperature
- `nSites`: number of lattice sites

Keyword arguments:
- `tol`: small number used to avoid division by zero in relative errors

Returns:
- Dictionary with keys: `"upup"`, `"dndn"`, `"updn"`, `"dnup"`, and `"errors"`
"""
function compute_pairing_correlators(un_up, vn_up,
    un_dn, vn_dn,
    E, T, nSites; tol=1e-12)

    nStates = length(E)
    @assert size(un_up, 1) == nStates == size(vn_up, 1) == size(un_dn, 1) == size(vn_dn, 1)
    @assert size(un_up, 2) == nSites == size(vn_up, 2) == size(un_dn, 2) == size(vn_dn, 2)

    Fuu = zeros(ComplexF64, nSites, nSites)
    Fdd = zeros(ComplexF64, nSites, nSites)
    Fud = zeros(ComplexF64, nSites, nSites)
    Fdu = zeros(ComplexF64, nSites, nSites)

    for n in 1:nStates
        w = 1 - 2 * f(E[n]; T=T)

        Fuu .+= w .* (un_up[n, :] * (vn_up[n, :])')
        Fdd .+= w .* (un_dn[n, :] * (vn_dn[n, :])')
        Fud .+= w .* (un_up[n, :] * (vn_dn[n, :])')
        Fdu .+= w .* (un_dn[n, :] * (vn_up[n, :])')
    end

    err_uu = maximum(abs.(Fuu .+ transpose(Fuu))) / max(maximum(abs.(Fuu)), tol)
    err_dd = maximum(abs.(Fdd .+ transpose(Fdd))) / max(maximum(abs.(Fdd)), tol)

    # singlet symmetric; triplet m0 antisymmetric
    Fs = 0.5 .* (Fud .- Fdu)
    Ft0 = 0.5 .* (Fud .+ Fdu)

    err_s = maximum(abs.(Fs .- transpose(Fs))) / max(maximum(abs.(Fs)), tol)   # should be ~0
    err_t0 = maximum(abs.(Ft0 .+ transpose(Ft0))) / max(maximum(abs.(Ft0)), tol)   # should be ~0

    return Dict(
        "upup" => Fuu,
        "dndn" => Fdd,
        "updn" => Fud,
        "dnup" => Fdu,
        "errors" => Dict("err_uu" => err_uu, "err_dd" => err_dd, "err_s" => err_s, "err_t0" => err_t0)
    )
end


"""
    triplet_diagnostics(F, neighbors)

Bond-resolved odd-parity triplet diagnostics from correlators.

Returns:
- rms_uu, rms_dd, rms_eqspin (uu+dd), rms_mz0, rms_singlet
- bond_map: (i,j,w_eqspin,w_mz0) for j>i bonds
"""
function triplet_diagnostics(F, neighbors)
    Fuu = F["upup"]
    Fdd = F["dndn"]
    Fud = F["updn"]
    Fdu = F["dnup"]

    nSites = length(neighbors)

    acc_uu = 0.0
    acc_dd = 0.0
    acc_eq = 0.0
    acc_0 = 0.0
    acc_s = 0.0
    nbonds = 0

    bond_map = Vector{Tuple{Int,Int,Float64,Float64}}()

    for i in 1:nSites
        for j in neighbors[i]
            if j > i
                # odd-parity equal-spin components
                fuu = (Fuu[i, j] - Fuu[j, i]) / sqrt(2)
                fdd = (Fdd[i, j] - Fdd[j, i]) / sqrt(2)

                w_uu = abs2(fuu)
                w_dd = abs2(fdd)
                w_eq = w_uu + w_dd

                # odd-parity m_z=0 triplet: antisym part of (Fud+Fdu)/2
                f0 = ((Fud[i, j] + Fdu[i, j]) - (Fud[j, i] + Fdu[j, i])) / (2 * sqrt(2))
                w_0 = abs2(f0)

                # singlet on the same bond (symmetric part of (Fud-Fdu)/2)
                fs = ((Fud[i, j] - Fdu[i, j]) + (Fud[j, i] - Fdu[j, i])) / (2 * sqrt(2))
                w_s = abs2(fs)

                acc_uu += w_uu
                acc_dd += w_dd
                acc_eq += w_eq
                acc_0 += w_0
                acc_s += w_s
                nbonds += 1

                push!(bond_map, (i, j, w_eq, w_0))
            end
        end
    end

    denom = nbonds > 0 ? nbonds : 1
    return (
        rms_uu=nbonds > 0 ? sqrt(acc_uu / denom) : 0.0,
        rms_dd=nbonds > 0 ? sqrt(acc_dd / denom) : 0.0,
        rms_eqspin=nbonds > 0 ? sqrt(acc_eq / denom) : 0.0,
        rms_mz0=nbonds > 0 ? sqrt(acc_0 / denom) : 0.0,
        rms_singlet=nbonds > 0 ? sqrt(acc_s / denom) : 0.0,
        bond_map=bond_map
    )
end

end # module end
