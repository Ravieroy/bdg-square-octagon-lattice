module BdGUtilities
    using LatticeUtilities
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

    # Module exports
    export sort_evecs
    export sort_evecs_spinfull
    export check_norm
    export check_norm_spinful
    export generate_t_matrix
    export calc_delta
    export calc_delta_spinfull
    export calc_avg_n
    export calc_avg_n_spinfull
    export create_bdg_ham_up
    export create_bdg_ham_dn
    export create_bdg_ham_full
    export add_rashba_soc!
    export add_intrinsic_soc!
    export compute_iso_map
    export generate_t_map
    export run_self_consistency_numpy
    export run_self_consistency_numpy_spinfull

    # alias to fermi function
    const f = fermi_fn

    """
        sort_evecs(evector::AbstractMatrix{T}, nSites::Int) where T <: Union{Float64, ComplexF64}

    Sorts and slices an eigenvector matrix `evector` to separate components `uₙ` and `vₙ`.

    # Arguments
    - `evector::AbstractMatrix{T}`: Matrix of eigenvectors (can be real or complex).
    - `nSites::Int`: Number of lattice sites.

    # Returns
    - Tuple of matrices (`uₙ`, `vₙ`), each with element type `T`.
    """
    function sort_evecs(evector::AbstractMatrix{T}, nSites::Int) where T <: Union{Float64, ComplexF64}
        # Transpose to rearrange the vectors for easier slicing
        evecs = transpose(evector)[nSites + 1 : 2 * nSites, :]

        uₙ = evecs[:, 1:nSites]
        vₙ = evecs[:, nSites + 1:2 * nSites]

        return uₙ, vₙ
    end

    """
        sort_evecs_spinfull(evector::AbstractMatrix{T}, nSites::Int) where T <: Union{Float64, ComplexF64}

    Sorts and slices an eigenvector matrix `evector` to separate components `uₙ↑`, `vₙ↓`, `uₙ↓`, `vₙ↑`.

    # Arguments
    - `evector::AbstractMatrix{T}`: Matrix of eigenvectors (can be real or complex).
    - `nSites::Int`: Number of lattice sites.

    # Returns
    - Tuple of matrices (`uₙ↑`, `vₙ↓`, `uₙ↓`, `vₙ↑`), each with element type `T`.
    """
    function sort_evecs_spinfull(evecs::AbstractMatrix{T}, nSites::Int) where T <: Union{Float64, ComplexF64}
        totalStates = 4 * nSites
        halfStates = div(totalStates, 2)

        # Transpose to work with eigenvectors as rows
        evecs_t = transpose(evecs)

        # Only take the second half (positive-energy eigenvectors)
        pos_evecs = evecs_t[halfStates+1:end, :]  # shape: N × 4N

        u_up = pos_evecs[:, 1:nSites]
        v_dn = pos_evecs[:, nSites+1:2nSites]
        u_dn = pos_evecs[:, 2nSites+1:3nSites]
        v_up = pos_evecs[:, 3nSites+1:4nSites]

        return u_up, v_dn, u_dn, v_up
    end


    """
        check_norm(uₙ::AbstractMatrix{T}, vₙ::AbstractMatrix{T}, nSites::Int; digits::Int=5) where T <: Union{Float64, ComplexF64}

    Checks the normalization condition |uₙ|^2 + |vₙ|^2 = 1 at every site for
    each eigenvalue `n`, with optional rounding precision.

    # Arguments
    - `uₙ::AbstractMatrix{T}`
    - `vₙ::AbstractMatrix{T}`
    - `nSites::Int`: Number of lattice sites.
    - `digits::Int`: Number of decimal places to round the output to (default is 5).

    # Returns
    - `normList::Vector{Float64}`: A vector of norm values rounded to the specified precision.
    """
    function check_norm(uₙ::AbstractMatrix{T}, vₙ::AbstractMatrix{T}, nSites::Int; digits::Int=5) where T <: Union{Float64, ComplexF64}
        normList = zeros(Float64, nSites)
        for n in 1:nSites
            val = dot(uₙ[n, :], uₙ[n, :]) + dot(vₙ[n, :], vₙ[n, :])
            normList[n] = val
        end
        return round.(normList, digits=digits)
    end


    """
        check_norm_spinful(u_up::AbstractMatrix{T}, v_dn::AbstractMatrix{T}, u_dn::AbstractMatrix{T}, v_up::AbstractMatrix{T},
            nSites::Int; digits::Int=5) where T <: Union{Float64, ComplexF64}

    Checks the normalization condition ∑ₙ,σ (|uₙσ|² + |vₙσ|²) = 1 at every site for
        each eigenvalue `n`, with optional rounding precision.

    # Arguments
    - `uₙ↑::AbstractMatrix{T}`
    - `vₙ↓::AbstractMatrix{T}`
    - `uₙ↓::AbstractMatrix{T}`
    - `vₙ↑::AbstractMatrix{T}`
    - `nSites::Int`: Number of lattice sites.
    - `digits::Int`: Number of decimal places to round the output to (default is 5).

    # Returns
    - `normList::Vector{Float64}`: A vector of norm values rounded to the specified precision.
        """
    function check_norm_spinful(
        u_up::AbstractMatrix{T}, v_dn::AbstractMatrix{T},
        u_dn::AbstractMatrix{T}, v_up::AbstractMatrix{T},
        nSites::Int; digits::Int=5) where T <: Union{Float64, ComplexF64}

        normList = zeros(Float64, nSites)
        for n in 1:nSites
            val = sum(abs2, u_up[n, :]) +
                  sum(abs2, v_dn[n, :]) +
                  sum(abs2, u_dn[n, :]) +
                  sum(abs2, v_up[n, :])
            normList[n] = val
        end
        return round.(normList, digits=digits)
    end


    """
        generate_t_matrix(fileName::String; save::Bool=false, outFile::String="tMat.csv")::Matrix{ComplexF64}

    Reads a real-valued text file with interleaved real and imaginary parts of a Hamiltonian,
    reconstructs the complex matrix, and optionally saves it. This file is generated from an
    external Fortran program.

    # Arguments
    - `fileName::String`: Path to the input file with rows of [Re, Im, Re, Im, ...].
    - `save::Bool`: Save the reconstructed matrix to `outFile` (default: `false`).
    - `outFile::String`: Output file name for saving the matrix (default: `"tMat.csv"`).

    # Returns
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
        calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, E₁, E₂, T)

    Calculates the superconducting gap Δ for each site.

    # Arguments
    - `U`: Interaction parameter.
    - `un_up`: `u` component for spin-up.
    - `vn_up`: `v` component for spin-up.
    - `un_dn`: `u` component for spin-down.
    - `vn_dn`: `v` component for spin-down.
    - `nSites`: Number of lattice sites.
    - `E₁`: Eigenvalues for the first set of states.
    - `E₂`: Eigenvalues for the second set of states.
    - `T`: Temperature.

    # Returns
    - `ΔList`: List of calculated Δ values for each site, which can be of type Float64 or ComplexF64.
    """
    function calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, E₁, E₂, T)
        ΔList = zeros(ComplexF64, nSites)
        for i in 1:nSites
            Δ = zero(ComplexF64)
            for n in 1:nSites
                E₁ₙ = E₁[n]
                E₂ₙ = E₂[n]
                fₙ₁ = f(E₁ₙ, T=T)
                fₙ₂ = f(E₂ₙ, T=T)
                Δ += U * ((un_up[n, i] * conj(vn_dn[n, i])) * (1 - fₙ₁) - (un_dn[n, i] * conj(vn_up[n, i])) * fₙ₂)
            end
            ΔList[i] = Δ
        end

        # Convert to Float64 if all values are real
        if all(isreal, ΔList)
            return Float64.(ΔList)
        end

        return ΔList
    end


    """
        calc_delta_spinfull(U, un_up, vn_up, un_dn, vn_dn, nSites, E, T)

    Calculates the superconducting gap Δ for each site for the spinfull case.

    # Arguments
    - `U`: Interaction parameter.
    - `un_up`: `u` component for spin-up.
    - `vn_up`: `v` component for spin-up.
    - `un_dn`: `u` component for spin-down.
    - `vn_dn`: `v` component for spin-down.
    - `nSites`: Number of lattice sites.
    - `E`: Eigenvalues.
    - `T`: Temperature.

    # Returns
    - `ΔList`: List of calculated Δ values for each site, which can be of type Float64 or ComplexF64.
    """
    function calc_delta_spinfull(U, un_up, vn_up, un_dn, vn_dn, nSites, E, T)
        ΔList = zeros(ComplexF64, nSites)
        for i in 1:nSites
            Δ = zero(ComplexF64)
            for n in 1:2nSites
                Eₙ = E[n]
                fₙ = f(Eₙ, T=T)
                Δ += U * ((un_up[n, i] * conj(vn_dn[n, i])) * (1 - fₙ) - (un_dn[n, i] * conj(vn_up[n, i])) * fₙ)
            end
            ΔList[i] = Δ
        end

        # Convert to Float64 if all values are real
        if all(isreal, ΔList)
            return Float64.(ΔList)
        end

        return ΔList
    end


    """
        calc_avg_n(un, vn, nSites, E₁, E₂, T)

    Calculates the superconducting gap Δ for each site.

    # Arguments
    - `un`: `u` component.
    - `vn`: `v` component.
    - `nSites`: Number of lattice sites.
    - `E₁`: Eigenvalues for the first set of states.
    - `E₂`: Eigenvalues for the second set of states.
    - `T`: Temperature.

    # Returns
    - `n̄List`: List of calculated average value of number density of electrons(n̄) for each site.
    """
    function calc_avg_n(un, vn, nSites, E₁, E₂, T)
        n̄List = zeros(Float64, nSites)
        for i in 1:nSites
            n̄ = 0
            for n in 1:nSites
                E₁ₙ = E₁[n]
                E₂ₙ = E₂[n]
                fₙ₁ = f(E₁ₙ, T=T)
                fₙ₂ = f(E₂ₙ, T=T)
                n̄ += (abs2(un[n, i]) * fₙ₁ + abs2(vn[n, i]) * (1 - fₙ₂))
            end
            n̄List[i] = n̄
        end
        return n̄List
    end


    """
        calc_avg_n_spinfull(un, vn, nSites, E, T)

    Calculates the superconducting gap Δ for each site for the spinfull case.

    # Arguments
    - `un`: `u` component.
    - `vn`: `v` component.
    - `nSites`: Number of lattice sites.
    - `E`: Eigenvalues.
    - `T`: Temperature.

    # Returns
    - `n̄List`: List of calculated average value of number density of electrons(n̄) for each site.
    """
    function calc_avg_n_spinfull(un, vn, nSites, E, T)
        n̄List = zeros(Float64, nSites)
        for i in 1:nSites
            n̄ = 0
            for n in 1:2nSites
                Eₙ = E[n]
                fₙ = f(Eₙ, T=T)
                n̄ += (abs2(un[n, i]) * fₙ + abs2(vn[n, i]) * (1-fₙ))
            end
            n̄List[i] = n̄
        end
        return n̄List
    end


    """
        create_bdg_ham_up(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J; K=0, isComplex=false)

    Constructs the Bogoliubov–de Gennes (BdG) Hamiltonian matrix for the spin-up sector,
    including on-site interactions, chemical potential, pairing terms, and magnetic impurity. This is
    a matrix which can be solved in absence of spin mixing, i.e., non-spinfull case.

    # Arguments
    - `deltaList::AbstractVector`: Superconducting order parameter Δᵢ at each site.
    - `H::AbstractMatrix`: Nearest neighbor matrix with hopping parameter t on the NN sites.
    - `μ::Real`: Chemical potential.
    - `nSites::Int`: Number of lattice sites.
    - `n_up::AbstractVector`: Spin-up occupation numbers ⟨nᵢ↑⟩.
    - `n_dn::AbstractVector`: Spin-down occupation numbers ⟨nᵢ↓⟩.
    - `U::Real`: On-site interaction strength.
    - `impuritySite::Int`: Site index of magnetic impurity.
    - `J::Real`: Exchange coupling for magnetic impurity.
    - `K::Real`: Magnetic anisotropy term (default: 0).
    - `isComplex::Bool`: Whether to construct the BdG Hamiltonian as a complex matrix (default: `false`).

    # Returns
    - `HBdG::AbstractMatrix`: The 2n × 2n BdG Hamiltonian matrix.
    """
    function create_bdg_ham_up(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J; K=0, isComplex=false)
        # Spin up and spin down indices
        σ_up = 1    # Spin-up
        σ_dn = -1   # Spin-down

        # Initialize the BdG Hamiltonian
        if isComplex
            HBdG = zeros(Complex{Float64}, 2 * nSites, 2 * nSites)
        else
            HBdG = zeros(Float64, 2 * nSites, 2 * nSites)
        end

        #----- Upper-left block: Spin-up sector------
        for i in 1:nSites
            if i == impuritySite
                # H[i, i] = -(μ + (U * n_dn[i]) + (K - σ_up * J)) # Hartree shift
                H[i, i] = -(μ + (K - σ_up * J))
            else
                # H[i, i] = -(μ + (U * n_dn[i]))
                H[i, i] = -μ
            end
        end

        for i in 1:nSites
            for j in 1:nSites
                HBdG[i, j] = H[i, j]
                HBdG[nSites + i, nSites + j] = -conj(H[i, j])
            end
        end

        # Add pairing terms D_ij in the off-diagonal
        for i in 1:nSites
            HBdG[i, nSites + i] = deltaList[i]          # D_ij in upper-left block
            HBdG[nSites + i, i] = conj(deltaList[i])    # D_ij^* in the upper-left block
        end

        #----- Upper-left block: Spin-down sector------
        for i in 1:nSites
            if i == impuritySite
                # HBdG[nSites + i, nSites + i] = (μ + (U * n_up[i]) + (K - σ_dn * J))
                HBdG[nSites + i, nSites + i] = (μ + (K - σ_dn * J))
            else
                # HBdG[nSites + i, nSites + i] = (μ + (U * n_up[i]))
                HBdG[nSites + i, nSites + i] = μ
            end
        end
        return HBdG
    end


    """
        create_bdg_ham_dn(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J; K=0, isComplex=false)

    Constructs the Bogoliubov–de Gennes (BdG) Hamiltonian matrix for the spin-down sector,
    including on-site interactions, chemical potential, pairing terms, and magnetic impurity. This is
    a matrix which can be solved in absence of spin mixing, i.e., non-spinfull case.

    # Arguments
    - `deltaList::AbstractVector`: Superconducting order parameter Δᵢ at each site.
    - `H::AbstractMatrix`: Nearest neighbor matrix with hopping parameter t on the NN sites.
    - `μ::Real`: Chemical potential.
    - `nSites::Int`: Number of lattice sites.
    - `n_up::AbstractVector`: Spin-up occupation numbers ⟨nᵢ↑⟩.
    - `n_dn::AbstractVector`: Spin-down occupation numbers ⟨nᵢ↓⟩.
    - `U::Real`: On-site interaction strength.
    - `impuritySite::Int`: Site index of magnetic impurity.
    - `J::Real`: Exchange coupling for magnetic impurity.
    - `K::Real`: Magnetic anisotropy term (default: 0).
    - `isComplex::Bool`: Whether to construct the BdG Hamiltonian as a complex matrix (default: `false`).

    # Returns
    - `HBdG::AbstractMatrix`: The 2n × 2n BdG Hamiltonian matrix.
    """
    function create_bdg_ham_dn(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J; K=0, isComplex=false)
        # Spin up and spin down indices
        σ_up = 1    # Spin-up
        σ_dn = -1   # Spin-down

        # Initialize the BdG Hamiltonian
        if isComplex
            HBdG = zeros(Complex{Float64}, 2 * nSites, 2 * nSites)
        else
            HBdG = zeros(Float64, 2 * nSites, 2 * nSites)
        end

        for i in 1:nSites
            if i == impuritySite
                # H[i, i] = -(μ + (U * n_up[i]) + (K - σ_dn * J))
                H[i, i] = -(μ + (K - σ_dn * J))
            else
                # H[i, i] = -(μ + (U * n_up[i]))
                H[i, i] = -μ
            end
        end

        for i in 1:nSites
            for j in 1:nSites
                HBdG[i, j] = H[i, j]
                HBdG[nSites + i, nSites + j] = -conj(H[i, j])
            end
        end

        # Add pairing terms D_ij in the off-diagonal
        for i in 1:nSites
            HBdG[i, nSites + i] = deltaList[i]          # D_ij in upper-left block
            HBdG[nSites + i, i] = conj(deltaList[i])    # D_ij^* in the upper-left block
        end

        for i in 1:nSites
            if i == impuritySite
                # HBdG[nSites + i, nSites + i] = (μ + (U * n_dn[i]) + (K - σ_up * J))
                HBdG[nSites + i, nSites + i] = (μ + (K - σ_up * J))
            else
                # HBdG[nSites + i, nSites + i] = (μ + (U * n_dn[i]))
                HBdG[nSites + i, nSites + i] = μ
            end
        end
        return HBdG
    end


    """
        create_bdg_ham_full(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J, x, y, neighbors, λ, t_map;
        K=0, isComplex=true, iso_map=nothing, λ_iso=0.0)

    Constructs the full 4n × 4n Bogoliubov–de Gennes Hamiltonian including both spin sectors,
    Rashba spin–orbit coupling, and optional intrinsic spin–orbit coupling.

    # Arguments
    - `deltaList::AbstractVector`: Superconducting order parameter Δᵢ at each site.
    - `H::AbstractMatrix`: Initial single-particle Hamiltonian matrix.
    - `μ::Real`: Chemical potential.
    - `nSites::Int`: Number of lattice sites.
    - `n_up::AbstractVector`: Spin-up occupation numbers ⟨nᵢ↑⟩.
    - `n_dn::AbstractVector`: Spin-down occupation numbers ⟨nᵢ↓⟩.
    - `U::Real`: On-site interaction strength.
    - `impuritySite::Int`: Site index of magnetic impurity (1-based).
    - `J::Real`: Exchange coupling for the magnetic impurity.
    - `x, y::AbstractVector`: Coordinates of each site.
    - `neighbors::Vector{Vector{Int}}`: List of nearest neighbors for each site.
    - `λ::Real`: Rashba spin–orbit coupling strength.
    - `t_map::AbstractMatrix`: Hopping matrix elements.
    - `K::Real`: Magnetic anisotropy term (default: 0).
    - `isComplex::Bool`: Construct complex BdG matrix if `true` (default: `true`).
    - `iso_map`: Mapping for intrinsic SOC terms (default: `nothing`).
    - `λ_iso::Real`: Intrinsic spin–orbit coupling strength (default: 0.0).

    # Returns
    - `Hfull::AbstractMatrix`: The 4n × 4n full BdG Hamiltonian matrix with SOC terms included.
    """
    function create_bdg_ham_full(deltaList, H, μ, nSites, n_up, n_dn, U, impuritySite, J, x, y, neighbors, λ, t_map;
                                 K=0, isComplex=true, iso_map=nothing, λ_iso=0.0)
        M = 2 * nSites
        T = isComplex ? Complex{Float64} : Float64
        Hfull = zeros(T, 2*M, 2*M)

        Hup = create_bdg_ham_up(deltaList, copy(H), μ, nSites, n_up, n_dn, U, impuritySite, J; K=K, isComplex=isComplex)
        Hdn = create_bdg_ham_dn(deltaList, copy(H), μ, nSites, n_up, n_dn, U, impuritySite, J; K=K, isComplex=isComplex)

        Hfull[1:M,       1:M      ] .= Hup
        Hfull[M+1:end,   M+1:end  ] .= Hdn

        add_rashba_soc!(Hfull, x, y, neighbors, λ; t_map=t_map)

        if iso_map !== nothing && λ_iso != 0.0
            add_intrinsic_soc!(Hfull, iso_map, λ_iso)
        end

        return Hfull
    end


    """
        add_rashba_soc!(Hfull::AbstractMatrix, x::AbstractVector, y::AbstractVector, neighbors::AbstractMatrix{<:Integer}, λ::Real; t_map::Union{Dict{Tuple{Int,Int},Float64},Nothing}=nothing)

    Adds Rashba spin–orbit coupling terms to the full 4n × 4n BdG Hamiltonian in-place.

    # Arguments
    - `Hfull::AbstractMatrix`: Full BdG Hamiltonian to modify (size must be 4n × 4n).
    - `x, y::AbstractVector`: Real-space coordinates of lattice sites.
    - `neighbors::AbstractMatrix{<:Integer}`: Neighbor indices for each site (size n × z).
    - `λ::Real`: Rashba spin–orbit coupling strength.
    - `t_map::Union{Dict, Nothing}`: Optional hopping amplitude map; if omitted, assumes uniform t=1.

    # Returns
    - `nothing`: The function modifies `Hfull` in-place.
    """

    function add_rashba_soc!(Hfull::AbstractMatrix,
         x::AbstractVector,
         y::AbstractVector,
         neighbors::AbstractMatrix{<:Integer},
         λ::Real;
         t_map::Union{Dict{Tuple{Int,Int},Float64},Nothing}=nothing)

        n = length(x)
        @assert size(Hfull,1) == 4n

        for i in 1:n
            xi, yi = x[i], y[i]
            for j in neighbors[i, :]
                dx, dy = x[j] - xi, y[j] - yi
                t_ij = t_map === nothing ? 1.0 : get(t_map, (i, j), get(t_map, (j, i), 1.0))

                S = -im * λ * t_ij * (-dy - im * dx)

                Hfull[i,       2n + j]     += S
                Hfull[2n + j,  i     ]     += conj(S)

                Hfull[n + i,   3n + j]     += conj(S)
                Hfull[3n + j,  n + i ]     += S
            end
        end
        return nothing
    end


    """
        add_intrinsic_soc!(Hfull::AbstractMatrix, iso_map::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Float64}}}, λ_iso::Real)

    Adds intrinsic spin–orbit coupling (SOC) terms to the full 4n × 4n BdG Hamiltonian in-place.

    # Arguments
    - `Hfull::AbstractMatrix`: Full BdG Hamiltonian to modify (must have size 4n × 4n).
    - `iso_map::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Float64}}}`: Map specifying second-neighbor hopping paths; each key `(i, j)` corresponds to a hopping term with associated list of paths `(k, ν)` where `ν=±1` encodes the SOC sign determined by lattice chirality.
    - `λ_iso::Real`: Intrinsic SOC coupling strength.

    # Returns
    - `nothing`: The function modifies `Hfull` in-place.
    """
    function add_intrinsic_soc!(
        Hfull::AbstractMatrix,
        iso_map::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Float64}}},
        λ_iso::Real
    )
        n = size(Hfull, 1) ÷ 4
        for ((i, j), paths) in iso_map
            for (k, ν) in paths
                t_iso = im * λ_iso * ν

                # Up-up hopping (e_ij ⋅ σ_z = ν)
                Hfull[i, j] += t_iso
                Hfull[j, i] -= t_iso

                # Down-down hopping
                Hfull[n + i, n + j] -= t_iso
                Hfull[n + j, n + i] += t_iso

                # Particle-hole (BdG) symmetry
                Hfull[2n + j, 2n + i] -= conj(t_iso)
                Hfull[2n + i, 2n + j] += conj(t_iso)

                Hfull[3n + j, 3n + i] += conj(t_iso)
                Hfull[3n + i, 3n + j] -= conj(t_iso)
            end
        end
    end


    """
        compute_iso_map(df::DataFrame)

    Computes the intrinsic SOC hopping map (iso_map) from a DataFrame containing lattice site coordinates
    and nearest/next-nearest neighbor indices, for use in intrinsic spin–orbit coupling calculations.

    # Arguments
    - `df::DataFrame`: DataFrame with columns:
        - `:siteIndex`: Unique integer ID for each lattice site.
        - `:x, :y`: Real-space coordinates of each site.
        - `:n1`–`:n4`: Indices of nearest neighbors.
        - `:n5`–`:n8`: Indices of next-nearest neighbors.

    # Returns
    - `iso_map::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Float64}}}`:
    A dictionary mapping each pair `(i, j)` of second-neighbor sites to a vector of paths `(k, ν)`,
    where `k` is the shared nearest neighbor site mediating the hopping and `ν=±1` is the sign of the
    SOC determined by the chirality of the path (via the z-component of the cross product).
    """
    function compute_iso_map(df::DataFrame)
        coords = Dict(row.siteIndex => [row.x, row.y] for row in eachrow(df))
        row_map = Dict(row.siteIndex => row for row in eachrow(df))

        nn_cols = [:n1, :n2, :n3, :n4]
        nnn_cols = [:n5, :n6, :n7, :n8]

        z_cross(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) = a[1]*b[2] - a[2]*b[1]

        iso_map = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()

        for row in eachrow(df)
            i = row.siteIndex
            ri = coords[i]
            nn_i = [row[c] for c in nn_cols]

            for nnn_col in nnn_cols
                j = row[nnn_col]
                if j == -1 || j == i
                    continue
                end

                rj = coords[j]
                j_row = get(row_map, j, nothing)
                if isnothing(j_row)
                    continue
                end

                nn_j = [j_row[c] for c in nn_cols]

                for k in intersect(nn_i, nn_j)
                    rk = coords[k]
                    d_ik = ri .- rk
                    d_kj = rj .- rk
                    ν = sign(z_cross(d_kj, d_ik))

                    key = (i, j)
                    value = (k, ν)
                    push!(get!(iso_map, key, Vector{Tuple{Int, Float64}}()), value)
                end
            end
        end

        return iso_map
    end


    """
        generate_t_map(df::DataFrame, t2::Float64)

    Given a DataFrame `df` with neighbor columns n1, n2, n3, n4, and site positions x, y,
    compute the symmetric hopping dictionary t_map. n4 is the NNN site intracell.

    - `df`: DataFrame containing at least columns x, y, n1, n2, n3, n4.
    - `t2`: hopping value for 2nd nearest neighbor (n4).

    Returns a dictionary mapping (i, j) site pairs to hopping values.
    """
    function generate_t_map(df::DataFrame, t2::Float64)
        nSites = size(df, 1)
        t1 = 1.0
        neighbors = hcat(df.n1, df.n2, df.n3, df.n4)
        t_map = Dict{Tuple{Int,Int}, Float64}()

        for i in 1:nSites
            for (j_idx, j) in enumerate(neighbors[i, :])
                t_val = j_idx == 4 ? t2 : t1  # 4th entry is 2nd NN
                t_map[(i, j)] = t_val
                t_map[(j, i)] = t_val  # ensure symmetric
            end
        end

        return t_map
    end


    function run_self_consistency_numpy(deltaOld, μ, nSites, n_up, n_dn, tMat, U, J, impuritySite, T; tol=1e-5, maxCount=150, isComplexCalc=false)
        startTime = time()
        count = 0
        flag = false

        while flag == false
            count += 1
            H1 = create_bdg_ham_up(deltaOld, tMat, μ, nSites, n_up, n_dn, U, impuritySite, J, isComplex=isComplexCalc)
            H2 = create_bdg_ham_dn(deltaOld, tMat, μ, nSites, n_up, n_dn, U, impuritySite, J, isComplex=isComplexCalc)

            if !is_hermitian(H1) || !is_hermitian(H2)
                println("Error: Matrix not hermitian")
            end
            (evals1, evecs1) = np.linalg.eigh(H1)
            un_up, vn_dn = sort_evecs(evecs1, nSites)

            (evals2, evecs2) = np.linalg.eigh(H2)
            un_dn, vn_up = sort_evecs(evecs2, nSites)

            evalPostive1 = evals1[nSites + 1 : end]
            evalPostive2 = evals2[nSites + 1 : end]

            deltaNew = calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, evalPostive1, evalPostive2, T)
            n_up_new = calc_avg_n(un_up, vn_dn, nSites, evalPostive1, evalPostive2, T)
            n_dn_new = calc_avg_n(un_dn, vn_up, nSites, evalPostive1, evalPostive2, T)

            deltaFlag = check_rel_tol(deltaOld, deltaNew, tol=tol)
            if deltaFlag == true
                deltaFinal = deltaNew
                n_up_final = n_up_new
                n_dn_final = n_dn_new
                nAvg = n_up_final + n_dn_final
                isConverged = true
                endTime = round(time() - startTime, digits = 2)
                return deltaFinal, n_up_final, n_dn_final, nAvg, evecs1, evecs2, evals1, evals2, isConverged, endTime, count

                break
            else
                deltaOld = deltaNew
                if count >= maxCount
                    deltaFinal = deltaNew
                    n_up_final = n_up_new
                    n_dn_final = n_dn_new
                    nAvg = n_up_final + n_dn_final
                    endTime = round(time() - startTime, digits = 2)
                    isConverged = false
                    flag = true
                    return deltaFinal, n_up_final, n_dn_final, nAvg, evecs1, evecs2, evals1, evals2, isConverged, endTime, count
                    break
                end
            end
        end
    end


    function run_self_consistency_numpy_spinfull(deltaOld, μ, nSites, n_up, n_dn, tMat, U, J, x, y, neighbors, λ, λ_iso, t_map, iso_map, impuritySite, T; tol=0.001, maxCount=100, isComplexCalc=false)
        startTime = time()
        count = 0
        flag = false

        while flag == false
            count += 1
            H_BdG = create_bdg_ham_full(deltaOld,
                 tMat,
                 μ,
                 nSites,
                 n_up,
                 n_dn,
                 U,
                 impuritySite,
                 J,
                 x, y,
                 neighbors,
                 λ,
                 t_map,
                 isComplex=isComplexCalc,
                 iso_map=iso_map,
                 λ_iso=λ_iso)

            # if !is_hermitian(H_BdG)
            #     logFileName = "../logs/error.log"
            #     message = string("Error: Matrix not hermitian")
            #     write_log(logFileName, "SUCCESS", message)
            # end

            (evals, evecs) = np.linalg.eigh(H_BdG)
            un_up, vn_dn, un_dn, vn_up = sort_evecs_spinfull(evecs, nSites)
            evalPostive = evals[2 * nSites + 1 : end]
            deltaNew = calc_delta_spinfull(U, un_up, vn_up, un_dn, vn_dn, nSites, evalPostive, T)

            n_up_new = calc_avg_n_spinfull(un_up, vn_dn, nSites, evalPostive, T)
            n_dn_new = calc_avg_n_spinfull(un_dn, vn_up, nSites, evalPostive, T)

            deltaFlag = check_rel_tol(deltaOld, deltaNew, tol=tol)
            if deltaFlag == true
                deltaFinal = deltaNew
                n_up_final = n_up_new
                n_dn_final = n_dn_new
                nAvg = n_up_final + n_dn_final
                isConverged = true
                endTime = round(time() - startTime, digits = 2)
                return deltaFinal, n_up_final, n_dn_final, nAvg, evecs, evals, isConverged, endTime, count
            else
                deltaOld = deltaNew
                n_up = n_up_new
                n_dn = n_dn_new
                if count >= maxCount
                    deltaFinal = deltaNew
                    n_up_final = n_up_new
                    n_dn_final = n_dn_new
                    nAvg = n_up_final + n_dn_final
                    endTime = round(time() - startTime, digits = 2)
                    isConverged = false
                    flag = true
                    return deltaFinal, n_up_final, n_dn_final, nAvg, evecs, evals, isConverged, endTime, count
                end
            end
        end
    end

end # Module end
