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
    export compute_pairing_correlators
    export triplet_rms
    export singlet_amplitude
    export normalized_triplet
    export extract_geometry
    export build_H_twisted
    export bdg_free_energy
    export create_bdg_ham_full_twisted
    export run_self_consistency_numpy_spinfull_twisted
    export bdg_spectrum_twist
    export free_energy_twist
    export compute_Ds_helicity
    export estimate_TBKT

    # alias to fermi function
    const f = fermi_fn

    """
    sort_evecs(evector, nSites)

    Sort and slice an eigenvector matrix to separate `un` and `vn` components.

    Arguments:
    - `evector`: matrix of eigenvectors (real or complex)
    - `nSites`: number of lattice sites

    Returns:
    - `(un, vn)`: tuple of matrices containing particle and hole components
    """
    function sort_evecs(evector::AbstractMatrix{<:Number}, nSites::Int)

        # Transpose to rearrange eigenvectors for easier slicing
        evecs = transpose(evector)[nSites + 1 : 2*nSites, :]

        un = evecs[:, 1:nSites]
        vn = evecs[:, nSites + 1 : 2*nSites]

        return un, vn
    end

    """
    sort_evecs_spinfull(evector, nSites)

    Sort and slice a spinful BdG eigenvector matrix into
    (u_up, v_dn, u_dn, v_up) components.

    Arguments:
    - `evector`: eigenvector matrix (real or complex)
    - `nSites`: number of lattice sites

    Returns:
    - `(u_up, v_dn, u_dn, v_up)` matrices
    """
    function sort_evecs_spinfull(
        evector::AbstractMatrix{<:Number},
        nSites::Int
        )
        totalStates = 4 * nSites
        halfStates  = div(totalStates, 2)

        evecsTranspose = transpose(evector)
        evecsPositive = evecsTranspose[halfStates+1:end, :]

        u_up = evecsPositive[:, 1:nSites]
        v_dn = evecsPositive[:, nSites+1 : 2*nSites]
        u_dn = evecsPositive[:, 2*nSites+1 : 3*nSites]
        v_up = evecsPositive[:, 3*nSites+1 : 4*nSites]

        return u_up, v_dn, u_dn, v_up
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
        calc_delta(U, un_up, vn_up, un_dn, vn_dn, nSites, E1, E2, T)

    Compute the superconducting gap Delta at each lattice site.

    Arguments:
    - `U`: interaction strength
    - `un_up`, `vn_up`: spin-up BdG components
    - `un_dn`, `vn_dn`: spin-down BdG components
    - `nSites`: number of lattice sites
    - `E1`, `E2`: eigenvalue arrays
    - `T`: temperature

    Returns:
    - vector of Delta values (real or complex)
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

        for i in 1:nSites
            Delta = zero(ComplexF64)
            for n in 1:nSites
                f1 = f(E1[n]; T=T)
                f2 = f(E2[n]; T=T)

                Delta += U * (
                    un_up[n, i] * conj(vn_dn[n, i]) * (1 - f1)
                    - un_dn[n, i] * conj(vn_up[n, i]) * f2
                    )
            end
            DeltaList[i] = Delta
        end

        return all(isreal, DeltaList) ? Float64.(DeltaList) : DeltaList
    end



    """
        calc_delta_spinfull(U, un_up, vn_up, un_dn, vn_dn, nSites, E, T)

    Compute the superconducting gap Delta at each lattice site
    for the spinful BdG case.

    Arguments:
    - `U`: interaction strength
    - `un_up`, `vn_up`: spin-up BdG components
    - `un_dn`, `vn_dn`: spin-down BdG components
    - `nSites`: number of lattice sites
    - `E`: eigenvalues (length = 2*nSites)
    - `T`: temperature

    Returns:
    - vector of Delta values (real or complex)
    """
    function calc_delta_spinfull(
        U,
        un_up::AbstractMatrix{<:Number},
        vn_up::AbstractMatrix{<:Number},
        un_dn::AbstractMatrix{<:Number},
        vn_dn::AbstractMatrix{<:Number},
        nSites::Int,
        E::AbstractVector{<:Number},
        T
    )
        DeltaList = zeros(ComplexF64, nSites)

        for i in 1:nSites
            Delta = zero(ComplexF64)
            for n in 1:2*nSites
                fn = f(E[n]; T=T)

                Delta += U * (
                    un_up[n, i] * conj(vn_dn[n, i]) * (1 - fn)
                    - un_dn[n, i] * conj(vn_up[n, i]) * fn
                )
            end
            DeltaList[i] = Delta
        end

        return all(isreal, DeltaList) ? Float64.(DeltaList) : DeltaList
    end


    """
        calc_avg_n(un, vn, nSites, E1, E2, T)

    Calculate the average electron density at each lattice site.

    Arguments:
    - `un`: u component
    - `vn`: v component
    - `nSites`: number of lattice sites
    - `E1`: eigenvalues for the first set of states
    - `E2`: eigenvalues for the second set of states
    - `T`: temperature

    Returns:
    - `avgNList`: vector of average electron density values
    """
    function calc_avg_n(un, vn, nSites, E1, E2, T)
        avgNList = zeros(Float64, nSites)

        for i in 1:nSites
            avgN = 0.0
            for n in 1:nSites
                f1 = f(E1[n]; T=T)
                f2 = f(E2[n]; T=T)
                avgN += abs2(un[n, i]) * f1 +
                        abs2(vn[n, i]) * (1 - f2)
            end
            avgNList[i] = avgN
        end

        return avgNList
    end


    """
        calc_avg_n_spinfull(un, vn, nSites, E, T)

    Calculate the average electron density at each lattice site
    for the spinful BdG case.

    Arguments:
    - `un`: u component
    - `vn`: v component
    - `nSites`: number of lattice sites
    - `E`: eigenvalues
    - `T`: temperature

    Returns:
    - `avgNList`: vector of average electron density values
    """
    function calc_avg_n_spinfull(un, vn, nSites, E, T)
        avgNList = zeros(Float64, nSites)

        for i in 1:nSites
            avgN = 0.0
            for n in 1:2*nSites
                fn = f(E[n]; T=T)
                avgN += abs2(un[n, i]) * fn +
                        abs2(vn[n, i]) * (1 - fn)
            end
            avgNList[i] = avgN
        end

        return avgNList
    end


    """
        create_bdg_ham_up(deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
                        K=0, isComplex=false)

    Construct the Bogoliubov-de Gennes (BdG) Hamiltonian matrix for the spin-up sector,
    including on-site interactions, chemical potential, pairing terms, and magnetic impurity.
    This is a matrix which can be solved in absence of spin mixing, i.e., non-spinfull case.

    The hamiltonian is written in the basis
    (c₁↑, c₂↑, …, cₙ↑; c₁↓†, c₂↓†, …, cₙ↓†)ᵀ

    Arguments:
    `deltaList`: initial values of order parameter for initialisation
    `H`: tight binding matrix
    `mu`: chemical potential
    `nSites`: number of sites (4N^2)
    `nUp, nDn`: initial values of up and down electron at each sites
    `U`: Attractice Hubbard U
    `impuritySite`: site position of magnetic impurity (implemented but not checked)
    `J`: strenth of the magentic impurity at site `impuritySite` (implemented but not checked)
    `K`: (default = 0) site position of non-magnetic impurity (implemented but not checked)
    `isComplex`: (default = false) is the calculation complex (like non-zero SOC, twist etc.)
    """
    function create_bdg_ham_up(
        deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
        K=0, isComplex=false
    )
        # Spin up and spin down indices
        sigmaUp = 1    # Spin-up
        sigmaDn = -1   # Spin-down

        # Initialize the BdG Hamiltonian
        if isComplex
            HBdG = zeros(ComplexF64, 2 * nSites, 2 * nSites)
        else
            HBdG = zeros(Float64, 2 * nSites, 2 * nSites)
        end

        # ----- Upper-left block: Spin-up sector ------
        for i in 1:nSites
            if i == impuritySite
                # H[i, i] = -(mu + (U * nDn[i]) + (K - sigmaUp * J)) # Hartree shift
                H[i, i] = -(mu + (K - sigmaUp * J))
            else
                # H[i, i] = -(mu + (U * nDn[i]))
                H[i, i] = -mu
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

        # ----- Upper-left block: Spin-down sector ------
        for i in 1:nSites
            if i == impuritySite
                # HBdG[nSites + i, nSites + i] = (mu + (U * nUp[i]) + (K - sigmaDn * J))
                HBdG[nSites + i, nSites + i] = (mu + (K - sigmaDn * J))
            else
                # HBdG[nSites + i, nSites + i] = (mu + (U * nUp[i]))
                HBdG[nSites + i, nSites + i] = mu
            end
        end

        return HBdG
    end


    """
        create_bdg_ham_dn(deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
                    K=0, isComplex=false)

    Construct the Bogoliubov-de Gennes (BdG) Hamiltonian matrix for the spin-down sector,
    including on-site interactions, chemical potential, pairing terms, and magnetic impurity.
    This is a matrix which can be solved in absence of spin mixing, i.e., non-spinfull case.

    The hamiltonian is written in the basis
    (c₁↓, c₂↓, …, cₙ↓; c₁↑†, c₂↑†, …, cₙ↑†)


    Arguments:
    `deltaList`: initial values of order parameter for initialisation
    `H`: tight binding matrix
    `mu`: chemical potential
    `nSites`: number of sites (4N^2)
    `nUp, nDn`: initial values of up and down electron at each sites
    `U`: Attractice Hubbard U
    `impuritySite`: site position of magnetic impurity (implemented but not checked)
    `J`: strenth of the magentic impurity at site `impuritySite` (implemented but not checked)
    `K`: (default = 0) site position of non-magnetic impurity (implemented but not checked)
    `isComplex`: (default = false) is the calculation complex (like non-zero SOC, twist etc.)
    """
    function create_bdg_ham_dn(
        deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J;
        K=0, isComplex=false
    )
        # Spin up and spin down indices
        sigmaUp = 1    # Spin-up
        sigmaDn = -1   # Spin-down

        # Initialize the BdG Hamiltonian
        if isComplex
            HBdG = zeros(ComplexF64, 2 * nSites, 2 * nSites)
        else
            HBdG = zeros(Float64, 2 * nSites, 2 * nSites)
        end

        for i in 1:nSites
            if i == impuritySite
                # H[i, i] = -(mu + (U * nUp[i]) + (K - sigmaDn * J))
                H[i, i] = -(mu + (K - sigmaDn * J))
            else
                # H[i, i] = -(mu + (U * nUp[i]))
                H[i, i] = -mu
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
                # HBdG[nSites + i, nSites + i] = (mu + (U * nDn[i]) + (K - sigmaUp * J))
                HBdG[nSites + i, nSites + i] = (mu + (K - sigmaUp * J))
            else
                # HBdG[nSites + i, nSites + i] = (mu + (U * nDn[i]))
                HBdG[nSites + i, nSites + i] = mu
            end
        end

        return HBdG
    end


    """
        create_bdg_ham_full( deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap;
            K=0, isComplex=true, isoMap=nothing, lambdaIso=0.0
        )

    Construct the full 4n x 4n Bogoliubov-de Gennes Hamiltonian including both spin sectors,
    Rashba spin-orbit coupling, and optional intrinsic spin-orbit coupling.

    The hamiltonian is written in the basis
    (c₁↑, c₂↑, …, cₙ↑; c₁↓†, c₂↓†, …, cₙ↓†; c₁↓, c₂↓, …, cₙ↓; c₁↑†, c₂↑†, …, cₙ↑†)

    Arguments:
    - `deltaList`: superconducting order parameter at each site
    - `H`: tight binding Hamiltonian
    - `mu`: chemical potential
    - `nSites`: number of lattice sites
    - `nUp`: initial spin-up occupation numbers
    - `nDn`: initial spin-down occupation numbers
    - `U`: Attractive on-site interaction strength
    - `impuritySite`: index of magnetic impurity (1-based)
    - `J`: exchange coupling for magnetic impurity
    - `x`, `y`: coordinates of each lattice site
    - `neighbors`: list of nearest neighbors for each site
    - `lambda`: Rashba spin-orbit coupling strength
    - `tMap`: hopping matrix elements
    - `K`: non-magnetic impurity strength (not checked)
    - `isComplex`: construct complex BdG matrix if true (default true)
    - `isoMap`: mapping for intrinsic SOC terms (default nothing)
    - `lambdaIso`: intrinsic spin-orbit coupling strength (default 0.0)

    Returns:
    - `Hfull`: full 4n x 4n BdG Hamiltonian with SOC terms included
    """
    function create_bdg_ham_full(
        deltaList, H, mu, nSites, nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap;
        K=0, isComplex=true, isoMap=nothing, lambdaIso=0.0
    )
        M = 2 * nSites
        T = isComplex ? ComplexF64 : Float64
        Hfull = zeros(T, 2*M, 2*M)

        Hup = create_bdg_ham_up(
            deltaList, copy(H), mu, nSites, nUp, nDn, U, impuritySite, J;
            K=K, isComplex=isComplex
        )

        Hdn = create_bdg_ham_dn(
            deltaList, copy(H), mu, nSites, nUp, nDn, U, impuritySite, J;
            K=K, isComplex=isComplex
        )

        Hfull[1:M,         1:M       ] .= Hup
        Hfull[M+1:end,     M+1:end   ] .= Hdn

        add_rashba_soc!(Hfull, x, y, neighbors, lambda; tMap=tMap)

        if isoMap !== nothing && lambdaIso != 0.0
            add_intrinsic_soc!(Hfull, isoMap, lambdaIso)
        end

        return Hfull
    end


    """
        add_rashba_soc!(Hfull, x, y, neighbors, lambda;
            tMap=nothing)

    Add Rashba spin-orbit coupling terms to the full 4n x 4n BdG Hamiltonian in-place.
    The form is S = −iλtᵢⱼ (-dy -idx)

    Arguments:
    - `Hfull`: full BdG Hamiltonian (size must be 4*n x 4*n)
    - `x`, `y`: real-space coordinates of lattice sites
    - `neighbors`: neighbor indices for each site
    - `lambda`: Rashba spin-orbit coupling strength
    - `tMap`: optional hopping amplitude map; if omitted, assumes uniform t = 1

    Returns:
    - `nothing` (modifies `Hfull` in-place)
    """
    function add_rashba_soc!(
        Hfull::AbstractMatrix,
        x::AbstractVector,
        y::AbstractVector,
        neighbors::Vector{Vector{Int}},
        lambda::Real;
        tMap::Union{Dict{Tuple{Int,Int},Float64},Nothing}=nothing
    )
        n = length(x)
        @assert size(Hfull, 1) == 4 * n

        for i in 1:n
            xi, yi = x[i], y[i]

            for j in neighbors[i]
                dx = x[j] - xi
                dy = y[j] - yi

                # hopping amplitude (NN / NNN aware)
                tij = tMap === nothing ?
                    1.0 :
                    get(tMap, (i, j), get(tMap, (j, i), 1.0))

                # Rashba SOC matrix element
                S = -im * lambda * tij * (-dy - im * dx)

                Hfull[i,        2*n + j] += S
                Hfull[2*n + j,  i      ] += conj(S)

                Hfull[n + i,    3*n + j] += conj(S)
                Hfull[3*n + j,  n + i  ] += S
            end
        end

        return nothing
    end



    """
        add_intrinsic_soc!(Hfull, isoMap, lambdaIso)

    [UNTESTED, CAN CAUSE ERROR]
    Add intrinsic spin-orbit coupling (SOC) terms to the full 4n x 4n BdG Hamiltonian in-place.

    Arguments:
    - `Hfull`: full BdG Hamiltonian to modify (size must be 4*n x 4*n)
    - `isoMap`: map specifying second-neighbor hopping paths; each key (i, j)
    corresponds to a hopping term with associated list of paths (k, nu),
    where nu = +/- 1 encodes the SOC sign determined by lattice chirality
    - `lambdaIso`: intrinsic SOC coupling strength

    Returns:
    - `nothing` (modifies `Hfull` in-place)
    """
    function add_intrinsic_soc!(
        Hfull::AbstractMatrix,
        isoMap::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Float64}}},
        lambdaIso::Real
    )
        n = size(Hfull, 1) ÷ 4

        for ((i, j), paths) in isoMap
            for (k, nu) in paths
                tIso = im * lambdaIso * nu

                # Up-up hopping (e_ij . sigma_z = nu)
                Hfull[i, j] += tIso
                Hfull[j, i] -= tIso

                # Down-down hopping
                Hfull[n + i, n + j] -= tIso
                Hfull[n + j, n + i] += tIso

                # Particle-hole (BdG) symmetry
                Hfull[2*n + j, 2*n + i] -= conj(tIso)
                Hfull[2*n + i, 2*n + j] += conj(tIso)

                Hfull[3*n + j, 3*n + i] += conj(tIso)
                Hfull[3*n + i, 3*n + j] -= conj(tIso)
            end
        end

        return nothing
    end


    """
        compute_iso_map(df)

    [UNTESTED, CAN CAUSE ERROR]
    Compute the intrinsic SOC hopping map (isoMap) from a DataFrame containing
    lattice site coordinates and nearest / next-nearest neighbor indices.

    Arguments:
    - `df`: DataFrame with columns:
        - `:siteIndex`: lattice site
        - `:x`, `:y`: real-space coordinates of each site
        - `:r1`-`:r3`: indices of nearest neighbors
        - `:r4`: indices of next nearest neighbors
        - `:r5`-`:r8`: indices of 3rd nearest neighbors

    Returns:
    - `isoMap`: dictionary mapping each pair (i, j) of second-neighbor sites
    to a vector of paths (k, nu), where k is the shared nearest neighbor
    mediating the hopping and nu = +/- 1 is the SOC sign determined by
    lattice chirality
    """
    function compute_iso_map(df::DataFrame)
        coords = Dict(row.siteIndex => [row.x, row.y] for row in eachrow(df))
        rowMap = Dict(row.siteIndex => row for row in eachrow(df))

        nnCols  = [:r1, :r2, :r3, :r4]
        nnnCols = [:r5, :r6, :r7, :r8]

        z_cross(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}) =
            a[1] * b[2] - a[2] * b[1]

        isoMap = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Float64}}}()

        for row in eachrow(df)
            i = row.siteIndex
            ri = coords[i]
            nn_i = [row[c] for c in nnCols]

            for nnnCol in nnnCols
                j = row[nnnCol]
                if j == -1 || j == i
                    continue
                end

                rj = coords[j]
                jRow = get(rowMap, j, nothing)
                if isnothing(jRow)
                    continue
                end

                nn_j = [jRow[c] for c in nnCols]

                for k in intersect(nn_i, nn_j)
                    rk = coords[k]
                    d_ik = ri .- rk
                    d_kj = rj .- rk
                    nu = sign(z_cross(d_kj, d_ik))

                    key = (i, j)
                    value = (k, nu)
                    push!(get!(isoMap, key, Vector{Tuple{Int, Float64}}()), value)
                end
            end
        end

        return isoMap
    end


    """
        generate_t_map(df::DataFrame, t2::Float64)

    Given a DataFrame `df` with neighbor columns r1, r2, r3, r4, and site positions x, y,
    compute the symmetric hopping dictionary tMap. r4 is the NNN site intracell.

    - `df`: DataFrame containing at least columns x, y, r1, r2, r3, r4.
    - `t2`: hopping value for 2nd nearest neighbor (n4).

    Returns a dictionary mapping (i, j) site pairs to hopping values.
    """
    function generate_t_map(neighbors::Vector{Vector{Int}};
                        t1::Float64 = 1.0,
                        t2::Float64 = 1.0)

        nSites = length(neighbors)
        tMap = Dict{Tuple{Int,Int}, Float64}()

        for i in 1:nSites
            for (j_idx, j) in enumerate(neighbors[i])
                tVal = (j_idx == 4) ? t2 : t1   # r4 = NNN
                tMap[(i, j)] = tVal
                tMap[(j, i)] = tVal            # Hermitian
            end
        end

        return tMap
    end


    function run_self_consistency_numpy(
        deltaOld, mu, nSites, nUp, nDn, tMat, U, J, impuritySite, T;
        tol=1e-5, maxCount=150, isComplexCalc=false
        )
        startTime = time()
        count = 0
        flag = false

        while flag == false
            count += 1

            H1 = create_bdg_ham_up(
                deltaOld, tMat, mu, nSites, nUp, nDn, U, impuritySite, J;
                isComplex=isComplexCalc
                )
            H2 = create_bdg_ham_dn(
                deltaOld, tMat, mu, nSites, nUp, nDn, U, impuritySite, J;
                isComplex=isComplexCalc
                )

            if !is_hermitian(H1) || !is_hermitian(H2)
                println("Error: Matrix not hermitian")
            end

            (evals1, evecs1) = np.linalg.eigh(H1)
            un_up, vn_dn = sort_evecs(evecs1, nSites)

            (evals2, evecs2) = np.linalg.eigh(H2)
            un_dn, vn_up = sort_evecs(evecs2, nSites)

            evalPositive1 = evals1[nSites + 1:end]
            evalPositive2 = evals2[nSites + 1:end]

            deltaNew = calc_delta(
                U, un_up, vn_up, un_dn, vn_dn,
                nSites, evalPositive1, evalPositive2, T
                )

            nUpNew = calc_avg_n(un_up, vn_dn, nSites, evalPositive1, evalPositive2, T)
            nDnNew = calc_avg_n(un_dn, vn_up, nSites, evalPositive1, evalPositive2, T)

            deltaFlag = check_rel_tol(deltaOld, deltaNew, tol=tol)

            if deltaFlag == true
                deltaFinal = deltaNew
                nUpFinal = nUpNew
                nDnFinal = nDnNew
                nAvg = nUpFinal + nDnFinal
                isConverged = true
                endTime = round(time() - startTime, digits=2)

                return deltaFinal, nUpFinal, nDnFinal, nAvg,
                evecs1, evecs2, evals1, evals2,
                isConverged, endTime, count
            else
                deltaOld = deltaNew

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
        tol=0.001, maxCount=100, isComplexCalc=false
        )
        startTime = time()
        count = 0
        flag = false

        while flag == false
            count += 1

            H_BdG = create_bdg_ham_full(
                deltaOld,
                tMat,
                mu,
                nSites,
                nUp,
                nDn,
                U,
                impuritySite,
                J,
                x, y,
                neighbors,
                lambda,
                tMap;
                isComplex=isComplexCalc,
                isoMap=isoMap,
                lambdaIso=lambdaIso
                )

            (evals, evecs) = np.linalg.eigh(H_BdG)

            un_up, vn_dn, un_dn, vn_up = sort_evecs_spinfull(evecs, nSites)
            evalPositive = evals[2*nSites + 1:end]

            deltaNew = calc_delta_spinfull(
                U, un_up, vn_up, un_dn, vn_dn,
                nSites, evalPositive, T
                )

            nUpNew = calc_avg_n_spinfull(un_up, vn_dn, nSites, evalPositive, T)
            nDnNew = calc_avg_n_spinfull(un_dn, vn_up, nSites, evalPositive, T)

            deltaFlag = check_rel_tol(deltaOld, deltaNew, tol=tol)

            if deltaFlag == true
                deltaFinal = deltaNew
                nUpFinal = nUpNew
                nDnFinal = nDnNew
                nAvg = nUpFinal + nDnFinal
                isConverged = true
                endTime = round(time() - startTime, digits=2)

                return deltaFinal, nUpFinal, nDnFinal, nAvg,
                evecs, evals, isConverged, endTime, count
            else
                deltaOld = deltaNew
                nUp = nUpNew
                nDn = nDnNew

                if count >= maxCount
                    deltaFinal = deltaNew
                    nUpFinal = nUpNew
                    nDnFinal = nDnNew
                    nAvg = nUpFinal + nDnFinal
                    endTime = round(time() - startTime, digits=2)
                    isConverged = false
                    flag = true

                    return deltaFinal, nUpFinal, nDnFinal, nAvg,
                    evecs, evals, isConverged, endTime, count
                end
            end
        end
    end


    """
        Compute real-space pairing correlators from a converged spinful BdG solution.

    Arguments:
    - un_up, vn_up : BdG amplitudes for spin ↑
    - un_dn, vn_dn : BdG amplitudes for spin ↓
      (arrays of size 2nSites × nSites, indexed by eigenstate n and site i)
    - E            : BdG eigenvalues (length 2nSites, positive energies)
    - T            : temperature
    - nSites       : number of lattice sites

    Returns:
    Dictionary with pairing correlators F_{ij}.
    """
    function compute_pairing_correlators(un_up, vn_dn, un_dn, vn_up, E, T, nSites)

        F_upup = zeros(ComplexF64, nSites, nSites)
        F_dndn = zeros(ComplexF64, nSites, nSites)
        F_updn = zeros(ComplexF64, nSites, nSites)
        F_dnup = zeros(ComplexF64, nSites, nSites)

        for n in 1:2nSites   # positive-energy BdG states
            En = E[n]
            w  = 1 - 2*f(En, T=T)

            for i in 1:nSites, j in 1:nSites
                F_upup[i,j] += w * un_up[n,i] * conj(vn_up[n,j])
                F_dndn[i,j] += w * un_dn[n,i] * conj(vn_dn[n,j])
                F_updn[i,j] += w * un_up[n,i] * conj(vn_dn[n,j])
                F_dnup[i,j] += w * un_dn[n,i] * conj(vn_up[n,j])
            end
        end

        return Dict(
            "upup" => F_upup,
            "dndn" => F_dndn,
            "updn" => F_updn,
            "dnup" => F_dnup
        )
    end


    """
        Compute RMS equal-spin triplet amplitude on nearest-neighbor bonds.
    neighbors is an nSites × z matrix (z = number of neighbors, e.g. 4).
    """
    function triplet_rms(F, neighbors::Vector{Vector{Int}})
        F_upup = F["upup"]
        F_dndn = F["dndn"]

        nSites = length(neighbors)

        acc = 0.0
        nbonds = 0

        for i in 1:nSites
            for j in neighbors[i]
                if j > i   # avoid double counting
                    # antisymmetrized (odd parity)
                    fuu = 0.5 * (F_upup[i, j] - F_upup[j, i])
                    fdd = 0.5 * (F_dndn[i, j] - F_dndn[j, i])

                    acc += abs2(fuu) + abs2(fdd)
                    nbonds += 1
                end
            end
        end

        return nbonds > 0 ? sqrt(acc / nbonds) : 0.0
    end



    """
        singlet_amplitude(F, nSites)

    Compute average on-site singlet pairing amplitude.
    """
    function singlet_amplitude(F, nSites)
        F_updn = F["updn"]
        F_dnup = F["dnup"]

        acc = 0.0
        for i in 1:nSites
            psi_ii = 0.5 * (F_updn[i, i] - F_dnup[i, i])
            acc += abs(psi_ii)
        end

        return acc / nSites
    end



    """
        Compute triplet amplitude normalized by singlet amplitude.
    """
    function normalized_triplet(F, neighbors, nSites)
        T = triplet_rms(F, neighbors)
        S = singlet_amplitude(F, nSites)

        # println("S = $S, T = $T")
        return T / S
    end



    """
        extract_geometry(df)

    Arguments:
    - `df`: dataframe containing all the details of the square octagon lattice.

    Returns:
    - `x, y, neighbors` : Tuple containing x, y coordinates and neighbors of each site.
    """
    function extract_geometry(df; formatted::Bool=false)
        nSites = nrow(df)
        x = Vector{Float64}(df.x)
        y = Vector{Float64}(df.y)
        if formatted
            # NN: n1, n2, n3 ; NNN: n4
            neighbors = [Int[df.n1[i], df.n2[i], df.n3[i], df.n4[i]] for i in 1:nSites]
        else
            # NN: r1, r2, r3 ; NNN: r4
            neighbors = [Int[df.r1[i], df.r2[i], df.r3[i], df.r4[i]] for i in 1:nSites]
        end

        return x, y, neighbors
    end


    """
        build_H_twisted(x::Vector{Float64}, neighbors::Vector{Vector{Int}},
                            tMap::Dict{Tuple{Int,Int},Float64}, twist::Float64)

    Arguments:
    `x`: a vector containing the x-coordinates of site positions
    `neighbors`: neighbors of site i
    `tMap`: dictionary mapping (i, j) site pairs to hopping values.
    `twist`: the phase twist added to the NN and NNN sites
    """
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
                dx = x[j] - xi
                dx -= round(dx / Lx) * Lx   # minimal-image PBC

                tij = get(tMap, (i,j), 0.0)
                if tij != 0.0
                    phase = exp(1im * twist * dx / Lx)
                    H[i, j] += tij * phase
                end
            end
        end

        return H, Lx
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



    """
        create_bdg_ham_full_twisted(deltaList, mu, nSites,
        nUp, nDn, U, impuritySite, J, x, y, neighbors, lambda, tMap;
        twist=0.0, K=0, isComplex=true, isoMap=nothing, lambdaIso=0.0)

    Construct the full 4n x 4n twisted Bogoliubov-de Gennes Hamiltonian including both spin sectors,
    Rashba spin-orbit coupling, and optional intrinsic spin-orbit coupling.

    The hamiltonian is written in the basis
    (c₁↑, c₂↑, …, cₙ↑; c₁↓†, c₂↓†, …, cₙ↓†; c₁↓, c₂↓, …, cₙ↓; c₁↑†, c₂↑†, …, cₙ↑†)

    Arguments:
    - `deltaList`: superconducting order parameter at each site
    - `mu`: chemical potential
    - `nSites`: number of lattice sites
    - `nUp`: initial spin-up occupation numbers
    - `nDn`: initial spin-down occupation numbers
    - `U`: Attractive on-site interaction strength
    - `impuritySite`: index of magnetic impurity (1-based)
    - `J`: exchange coupling for magnetic impurity
    - `x`, `y`: coordinates of each lattice site
    - `neighbors`: list of nearest neighbors for each site
    - `lambda`: Rashba spin-orbit coupling strength
    - `tMap`: hopping matrix elements
    - `twist`: twist (default=0.0)
    - `K`: non-magnetic impurity strength (not checked)
    - `isComplex`: construct complex BdG matrix if true (default true)
    - `isoMap`: mapping for intrinsic SOC terms (default nothing)
    - `lambdaIso`: intrinsic spin-orbit coupling strength (default 0.0)

    Returns:
    - `Hfull`: full 4n x 4n BdG Hamiltonian with SOC terms included
    """
    function create_bdg_ham_full_twisted(deltaList, mu, nSites,
        nUp, nDn, U, impuritySite, J, x, y, neighbors, lambda, tMap;
        twist=0.0, K=0, isComplex=true, isoMap=nothing, lambdaIso=0.0)

        M = 2 * nSites
        T = isComplex ? ComplexF64 : Float64
        Hfull = zeros(T, 2*M, 2*M)

        H_delta, _ = build_H_twisted(x, neighbors, tMap, twist)

        Hup = create_bdg_ham_up(
            deltaList, copy(H_delta), mu, nSites,
            nUp, nDn, U, impuritySite, J;
            K=K, isComplex=isComplex
        )

        Hdn = create_bdg_ham_dn(
            deltaList, copy(H_delta), mu, nSites,
            nUp, nDn, U, impuritySite, J;
            K=K, isComplex=isComplex
        )

        Hfull[1:M,       1:M      ] .= Hup
        Hfull[M+1:end,   M+1:end  ] .= Hdn

        add_rashba_soc!(Hfull, x, y, neighbors, lambda; tMap=tMap)

        # if isoMap !== nothing && lambdaIso != 0.0
        #     add_intrinsic_soc!(Hfull, isoMap, lambdaIso)
        # end

        return Hfull
    end



    function run_self_consistency_numpy_spinfull_twisted(
        deltaOld, mu, nSites, nUp, nDn, U, J, x, y,
        neighbors, lambda, lambdaIso, tMap, isoMap, impuritySite, T;
        twist=0.0, K=0, tol=0.001, maxCount=200, isComplexCalc=false,
        verboseLogIn=nothing
        )
        startTime = time()
        count = 0
        flag = false

        while flag == false
            count += 1
            innerLoopTime = time()

            H_BdG = create_bdg_ham_full_twisted(
                deltaOld, mu, nSites,
                nUp, nDn, U, impuritySite, J,
                x, y, neighbors, lambda, tMap;
                twist=twist,
                K=K, isComplex=isComplexCalc
                )

            (evals, evecs) = np.linalg.eigh(H_BdG)

            un_up, vn_dn, un_dn, vn_up = sort_evecs_spinfull(evecs, nSites)
            evalPositive = evals[2*nSites + 1:end]

            deltaNew = calc_delta_spinfull(
                U, un_up, vn_up, un_dn, vn_dn,
                nSites, evalPositive, T
                )

            nUpNew = calc_avg_n_spinfull(un_up, vn_dn, nSites, evalPositive, T)
            nDnNew = calc_avg_n_spinfull(un_dn, vn_up, nSites, evalPositive, T)

            deltaFlag = check_rel_tol(deltaOld, deltaNew, tol=tol)

            if deltaFlag == true
                deltaFinal = deltaNew
                nUpFinal = nUpNew
                nDnFinal = nDnNew
                nAvg = nUpFinal + nDnFinal
                isConverged = true
                endTime = round(time() - startTime, digits=2)

                return deltaFinal, nUpFinal, nDnFinal, nAvg,
                evecs, evals, isConverged, endTime, count
            else
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
                    endTime = round(time() - startTime, digits=2)
                    isConverged = false
                    flag = true

                    return deltaFinal, nUpFinal, nDnFinal, nAvg,
                    evecs, evals, isConverged, endTime, count
                end
            end
        end
    end



    """
        bdg_spectrum_twist(
            deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap;
            twist=0.0, K=0, isComplexCalc=true
        )

    Compute the BdG eigenvalue spectrum with twisted boundary conditions.

    Arguments:
    - deltaList: Converged order parameter at all sites.
    - mu: Chemical potential.
    - nSites: Number of lattice sites.
    - nUp, nDn: Spin-up and spin-down particle numbers.
    - U: Attractive interaction strength.
    - impuritySite: Index of impurity (magnetic) site (not checked).
    - J: Impurity (magnetic) coupling strength.
    - x, y: Site coordinates.
    - neighbors: Neighbor list for hopping.
    - lambda: Rashba Spin-orbit coupling strength.
    - tMap: Hopping map.

    Keyword Arguments:
    - twist: Boundary condition twist angle.
    - K: Impurity (non-magnetic) coupling strength.
    - isComplexCalc: Use complex Hamiltonian if true.

    Returns:
    - evals: Array of BdG eigenvalues (full spectrum).
    """
    function bdg_spectrum_twist(
        deltaList, mu, nSites,
        nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap;
        twist=0.0, K=0, isComplexCalc=true
        )
        Hfull = create_bdg_ham_full_twisted(
            deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap;
            twist=twist,
            K=K, isComplex=isComplexCalc
            )

        (evals, evecs) = np.linalg.eigh(Hfull)
        return evals
    end


    """
        free_energy_twist(
            twist, deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap, T;
            K=0
        )

    Compute the BdG free energy with twisted boundary conditions.

    Arguments:
    - twist: twist angle.
    - deltaList: Converged order parameter at all sites.
    - mu: Chemical potential.
    - nSites: Number of lattice sites.
    - nUp, nDn: Spin-up and spin-down particle numbers.
    - U: Attractive interaction strength.
    - impuritySite: Index of impurity (magnetic) site (not checked).
    - J: Impurity (magnetic) coupling strength.
    - x, y: Site coordinates.
    - neighbors: Neighbor list for hopping.
    - lambda: Rashba spin-orbit coupling strength.
    - tMap: Hopping map.
    - T: Temperature.

    Keyword Arguments:
    - K: Impurity (non-magnetic) coupling strength.

    Returns:
    - Free energy computed from the full BdG spectrum.
    """

    function free_energy_twist(
        twist, deltaList, mu, nSites,
        nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap, T;
        K=0
        )
        evals = bdg_spectrum_twist(
            deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap;
            twist=twist, K=K, isComplexCalc=true
            )

        return bdg_free_energy(evals, deltaList, U, T)
    end

    """
        compute_Ds_helicity(
            deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap, T;
            K=0, twist=1e-3
        )

    Compute the superfluid stiffness using the helicity modulus.

    Arguments:
    - deltaList: Converged order parameter at all sites.
    - mu: Chemical potential.
    - nSites: Number of lattice sites.
    - nUp, nDn: Spin-up and spin-down particle numbers.
    - U: Attractive interaction strength.
    - impuritySite: Index of impurity (magnetic) site (not checked).
    - J: Impurity (magnetic) coupling strength.
    - x, y: Site coordinates.
    - neighbors: Neighbor list for hopping.
    - lambda: Rashba spin-orbit coupling strength.
    - tMap: Hopping map.
    - T: Temperature.

    Keyword Arguments:
    - K: Impurity (non-magnetic) coupling strength.
    - twist: Small boundary condition twist used for finite-difference derivative.

    Returns:
    - Ds: Superfluid stiffness (real part).
    """
    function compute_Ds_helicity(
        deltaList, mu, nSites,
        nUp, nDn, U, impuritySite, J,
        x, y, neighbors, lambda, tMap, T;
        K=0, twist=1e-3
        )
        xmin, xmax = minimum(x), maximum(x)
        Lx = xmax - xmin

        F0 = free_energy_twist(
            0.0, deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap, T; K=K
            )

        Fp = free_energy_twist(
            +twist, deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap, T; K=K
            )

        Fm = free_energy_twist(
            -twist, deltaList, mu, nSites,
            nUp, nDn, U, impuritySite, J,
            x, y, neighbors, lambda, tMap, T; K=K
            )

        Ds = (Lx^2 / nSites) * (Fp + Fm - 2*F0) / twist^2
        return real(Ds)
    end


    function estimate_TBKT(TVals, DsVals)
        for i in 1:length(TVals)-1
            f1 = DsVals[i]   - (2*TVals[i]   / pi)
            f2 = DsVals[i+1] - (2*TVals[i+1] / pi)

            if f1 * f2 <= 0   # crossing
                # linear interpolation
                T1, T2 = TVals[i], TVals[i+1]
                return T1 + (T2 - T1) * abs(f1) / (abs(f1) + abs(f2))
            end
        end
        return NaN
    end


end # Module end
