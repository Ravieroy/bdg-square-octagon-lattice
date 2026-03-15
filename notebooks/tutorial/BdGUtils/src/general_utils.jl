module GeneralUtils
    # Libraries required
    using JLD2, FileIO
    using HDF5, JLD
    using NPZ
    using PyCall
    using PyPlot
    using Dates
    using Crayons

    @pyimport pickle

    #Module exports
    export check_rel_tol
    export save_file
    export load_file
    export write_to_file
    export show_all
    export show_matrix
    export kronecker_delta
    export delta_fn
    export fermi_fn
    export fermi_fn_classic
    export get_present_time
    export write_to_file
    export create_folders_if_not_exist
    export is_symmetric
    export is_hermitian

    """
        check_rel_tol(oldList, newList; tol=1e-5, nRound=10)

    Check whether two vectors have converged within a relative tolerance.

    Arguments:
    - `oldList`: vector of previous values
    - `newList`: vector of updated values
    - `tol`: relative tolerance (default `1e-5`)
    - `nRound`: number of digits used for rounding relative differences

    Returns:
    - `true` if all elementwise relative differences are ≤ `tol`, otherwise `false`
    """
    function check_rel_tol(oldList::Vector{T}, newList::Vector{T}; tol::Real=1e-5, nRound::Int=10) where T
        @assert length(oldList) == length(newList) "Lists must be of the same length"

        # Calculate relative differences using magnitudes for both real and complex numbers
        relVals = abs.((newList .- oldList) ./ newList)
        relTolList = [round(i, digits=nRound) for i in relVals]
        return all(relTolList .<= tol)
    end



    """
        save_file(object, fileName; key="data")

    Save an object to disk in one of the supported file formats:
    `jld`, `jld2`, `npz`, or `pkl`.

    Arguments:
    - `object`: data object to be saved
    - `fileName`: output file name (extension determines format)
    - `key`: dataset name used for `jld`/`jld2` files (default `"data"`)

    Returns:
    - `nothing`
    """

    function save_file(object, fileName; key = "data")

        ext = lowercase(splitext(fileName)[2])  # ".jld", ".jld2", ".npz", ".pkl"

        if ext == ".jld"
            save(fileName, key, object)

        elseif ext == ".jld2"
        save(fileName, key, object)

        elseif ext == ".npz"
        npzwrite(fileName, object)

        elseif ext == ".pkl"
            open(fileName, "w") do f
                pickle.dump(object, f, protocol = pickle.HIGHEST_PROTOCOL)
            end

        else
            throw(ArgumentError(
                "Unsupported file format: $ext. Use jld, jld2, npz, or pkl."
                ))
        end

        return nothing
    end



    """
        load_file(fileName; key="data")

    Load and return data from a supported file format.

    Arguments:
    - `fileName`: input file name (extension determines format)
    - `key`: dataset name used for `jld`/`jld2` files (default `"data"`)

    Returns:
    - the loaded object
    """
    function load_file(fileName; key = "data")

        ext = lowercase(splitext(fileName)[2])

        if ext == ".npz"
            return npzread(fileName)

            elseif ext == ".jld" || ext == ".jld2"
            return load(fileName, key)

            elseif ext == ".pkl"
            return open(fileName, "r") do f
                pickle.load(f)
            end

        else
            throw(ArgumentError(
                "Unsupported file format: $ext. Use npz, jld, jld2, or pkl."
                ))
        end
    end


    """
        show_all(obj)

    Arguments:
    - `obj`: object which is truncated (like matrix, vector etc.)

    Display `obj` to standard output without truncation.
    """
    function show_all(obj)
        show(stdout, MIME("text/plain"), obj)
        return nothing
    end



    """
        show_matrix(mat)

    Arguments:
    - `mat`: Matrix

    Display a matrix as a grayscale image with no interpolation and a colorbar.
    """
    function show_matrix(mat)
        fig, ax = subplots()
        im = ax.imshow(mat, cmap="gray", interpolation="none")
        fig.colorbar(im, ax=ax)
        return nothing
    end


    """
        kronecker_delta(i, j)

    Returns 0 if i ≠ j and 1 if i == j
    """
    @inline kronecker_delta(i, j) = i == j ? one(Int) : zero(Int)


    """
        delta_fn(x, Γ)

    Calculates the delta function value at a given point.

    Arguments
    - `x`: The input value (floating-point number).
    - `Γ`: The broadening parameter (floating-point number).

    Returns
    - The delta function value as a floating-point number.
    """
    @inline function delta_fn(x, Γ)
        @assert Γ > 0 "Broadening Γ must be positive"
        return inv(pi) * Γ / (x^2 + Γ^2)
    end


    """
        fermi_fn(E, mu=0.0; beta=nothing, T=nothing)

    Evaluate the Fermi-Dirac distribution using a numerically stable tanh form.

    Arguments:
    - `E`: energy value
    - `mu`: chemical potential (default 0.0)
    - `beta`: inverse temperature (1/T)
    - `T`: temperature

    Returns:
    - Fermi-Dirac occupation value
    """
    @inline function fermi_fn(E, mu=0.0; beta=nothing, T=nothing)

        if beta === nothing
            T === nothing && throw(ArgumentError("Provide either `beta` or `T`."))
            beta = inv(T)
        end

        x = beta * (E - mu)
        return 0.5 * (1 - tanh(0.5 * x))
    end



    """
        fermi_fn_classic(E, T)

    Evaluate the Fermi-Dirac distribution using the exponential form.

    Arguments:
    - `E`: energy value
    - `T`: temperature

    Returns:
    - Fermi-Dirac occupation value
    """
    @inline function fermi_fn_classic(E, T)
        x = E / T
        if x > 50
            return zero(x)
            elseif x < -50
            return one(x)
        else
            return inv(exp(x) + one(x))
        end
    end


    """
        fermi_fn_old(E::Float64, μ::Float64=0.0; kwargs...)::Float64

    Evaluate the Fermi-Dirac distribution using a numerically stable tanh form
    given energy `E`, chemical potential `μ`. Accepts `β`, `beta`, or `T`.

    Arguments
    - `E::Float64`: Energy level.
    - `μ::Float64`: Chemical potential (default is `0.0`).
    - `kwargs...`: Optional keyword arguments, where one of `:β`, `:beta`, or `:T` must be specified.

    Returns
    - A `Float64` result based on the specified temperature parameter.
    """
    function fermi_fn_old(E::Float64, μ::Float64=0.0; kwargs...)::Float64
        if :β in keys(kwargs)
            β = kwargs[:β]
            return (1 - tanh(0.5 * β * (E - μ))) * 0.5
        elseif :beta in keys(kwargs)
            β = kwargs[:beta]
            return (1 - tanh(0.5 * β * (E - μ))) * 0.5
        elseif :T in keys(kwargs)
            T = kwargs[:T]
            return (1 - tanh(0.5 * (1 / T) * (E - μ))) * 0.5
        else
            throw(ArgumentError("Provide one of `β`, `beta`, or `T`."))
        end
    end



    """
        fermi_fn_classic(E, T)

    Returns the value of Fermi function using the classic formula.
    """
    function fermi_fn_classic_old(E, T)
        return 1 / (exp(E / T) + 1)
    end

    """
        get_present_time()

    Returns present time.
    """
    function get_present_time()
        timeNow = Dates.format(now(), "HH:MM:SS")
        dateToday = Dates.format(now(), DateFormat("d-m"))
        return timeNow, dateToday
    end

    """
        write_to_file(message, fileName; mode="a")

    Writes the message into the file and save it locally.
    """
    function write_to_file(message, fileName; mode="a")
        f = open(fileName, mode)
        write(f, message)
        close(f)
    end

    """
        create_folders_if_not_exist(folder_names::Vector{String}, base_path::AbstractString="")

    Creates folders if they do not exist
    """
    function create_folders_if_not_exist(folder_names::Vector{String}, base_path::AbstractString="")
        for folder_name in folder_names
            folder_path = joinpath(base_path, folder_name)
            if !isdir(folder_path)
                mkdir(folder_path)
                println(Crayon(foreground = :green, bold = true)("Folder created: $folder_path"))
            else
                println(Crayon(foreground = :yellow, bold = true)("Folder already exists: $folder_path"))
            end
        end
    end


    """
        is_symmetric(A; tol=1e-10)

    Check whether a real square matrix is symmetric within a tolerance.

    Arguments:
    - `A`: real square matrix
    - `tol`: tolerance for symmetry check

    Returns:
    - `true` if max(abs(A - transpose(A))) < tol, otherwise `false`
    """
    function is_symmetric(A::AbstractMatrix{<:Real}; tol::Real=1e-10)
        size(A, 1) == size(A, 2) ||
            throw(ArgumentError("Matrix must be square; got size=$(size(A))"))
        return maximum(abs.(A .- transpose(A))) < tol
    end


    """
        is_hermitian(A; tol=1e-10)

    Check whether a square matrix is Hermitian within a tolerance.

    Arguments:
    - `A`: real or complex square matrix
    - `tol`: tolerance for Hermiticity check

    Returns:
    - `true` if max(abs(A - adjoint(A))) < tol, otherwise `false`
    """
    function is_hermitian(A::AbstractMatrix{<:Number}; tol::Real=1e-10)
        size(A, 1) == size(A, 2) ||
            throw(ArgumentError("Matrix must be square; got size=$(size(A))"))
        return maximum(abs.(A .- adjoint(A))) < tol
    end

end #module end
