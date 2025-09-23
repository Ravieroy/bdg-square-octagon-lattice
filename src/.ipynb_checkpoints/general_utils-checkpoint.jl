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
    export create_folders_if_not_exist
    export is_symmetric

    # """
    #     check_rel_tol(oldList::Vector{T}, newList::Vector{T}; tol=1e-5, nRound=10)

    # Checks if the relative tolerance between two lists is within the specified tolerance.

    # # Arguments
    # - `oldList`: The original list of values.
    # - `newList`: The new list of values.
    # - `tol`: The tolerance level (default is `1e-5`).
    # - `nRound`: Number of digits to round the relative differences (default is `10`).

    # # Returns
    # - Boolean value indicating if all relative differences are within the specified tolerance.
    # """
    # function check_rel_tol(oldList::Vector{T}, newList::Vector{T}; tol::Real=1e-5, nRound::Int=10) where T
    #     @assert length(oldList) == length(newList) "Lists must be of the same length"

    #     # Calculate relative differences using magnitudes for both real and complex numbers
    #     relVals = abs.((newList .- oldList) ./ newList)
    #     relTolList = [round(i, digits=nRound) for i in relVals]
    #     return all(relTolList .<= tol)
    # end

    """
        check_rel_tol(oldList::Vector{T}, newList::Vector{T}; tol=1e-5, nRound=10)

    Checks if the relative tolerance between two lists is within the specified tolerance.

    # Arguments
    - `oldList`: The original list of values.
    - `newList`: The new list of values.
    - `tol`: The tolerance level (default is `1e-5`).
    - `nRound`: Number of digits to round the relative differences (default is `10`).

    # Returns
    - Boolean value indicating if all relative differences are within the specified tolerance.
    """
    function check_rel_tol(oldList::Vector{T}, newList::Vector{T}; tol::Real=1e-5, abs_tol::Real=1e-7, nRound::Int=10) where T
        @assert length(oldList) == length(newList) "Lists must be of the same length"

        # Compute relative differences
        relVals = abs.((newList .- oldList) ./ newList)

        # Use absolute tolerance for small values
        small_mask = abs.(newList) .< abs_tol
        relVals[small_mask] .= abs.(newList[small_mask] .- oldList[small_mask])

        # Round and check tolerance
        relTolList = [round(i, digits=nRound) for i in relVals]
        return all(relTolList .<= tol)
    end



    """
        save_file(object, fileName; key="data")
    Saves the given object in desired format(jld, jld2, pkl, npz)
    """
    function save_file(object, fileName; key = "data")
        if last(fileName, 3) == "jld"
            save(fileName, key, object)
        elseif last(fileName, 3) == "npz"
            npzwrite(fileName, object)
        elseif last(fileName, 4) == "jld2"
            save(fileName, key, object)
        elseif last(fileName, 3) == "pkl"
            f = open(fileName, "w")
            pickle.dump(object, f, protocol = pickle.HIGHEST_PROTOCOL)
            close(f)
        else
            throw("ERROR : Possibly wrong format ~ Try jld, jld2, npz or pkl")
        end

    end # save_file end

    """
        load_file(fileName; key="data")
    Loads the file from formats(npz, jld, jld2, pkl)
    """
    function load_file(fileName; key = "data")
        if last(fileName, 3) == "npz"
            mat = npzread(fileName)
        elseif last(fileName, 3) == "jld" || last(fileName, 4) == "jld2"
            mat = load(fileName)[key]
        elseif last(fileName, 3) == "pkl"
            # load the pickle file.
            f = open(fileName, "r")
            mat = pickle.load(f)
        else
            println("ERROR : $fileName not found")
        end
    end # load_file end



    """
        show_all(obj)
    shows the obj without any truncation
    """
    function show_all(obj)
        return show(stdout, "text/plain", obj)
    end

    function show_matrix(mat)
        PyPlot.gray()
        imshow(mat,interpolation="none")
        colorbar()
    end

    """
        kronecker_delta(i, j)
    Returns 0 if i ≠ j and 1 if i == j
    """
    function kronecker_delta(i, j)
        return i == j ?  1 : 0
    end


    """
        delta_fn(x, Γ)

    Calculates the delta function value at a given point.

    # Arguments
    - `x`: The input value (floating-point number).
    - `Γ`: The broadening parameter (floating-point number).

    # Returns
    - The delta function value as a floating-point number.
    """
    function delta_fn(x, Γ)
        return 1 / π * (Γ / (Γ^2 + x^2))
    end


    """
    f(E::Float64, μ::Float64=0.0; kwargs...)::Float64

    Calculates a thermodynamic quantity given energy `E`, chemical potential `μ`,
    and a temperature-related parameter. Accepts `β`, `beta`, or `T`.

    # Arguments
    - `E::Float64`: Energy level.
    - `μ::Float64`: Chemical potential (default is `0.0`).
    - `kwargs...`: Optional keyword arguments, where one of `:β`, `:beta`, or `:T` must be specified.

    # Returns
    - A `Float64` result based on the specified temperature parameter.
    """
    function fermi_fn(E::Float64, μ::Float64=0.0; kwargs...)::Float64
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
    function fermi_fn_classic(E, T)
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
    is_symmetric(matrix::AbstractMatrix{T}; tol::Real=1e-10):: Bool

    Checks if the given matrix is symmetric within a specified tolerance.

    # Arguments
    - `matrix`: A square matrix of type `AbstractMatrix{T}`, where `T` can be either `Real` or `Complex`.
    - `tol`: A real number representing the tolerance level for symmetry check (default is `1e-10`).

    # Returns
    - A boolean value indicating whether the matrix is symmetric.
    """
    function is_symmetric(matrix::AbstractMatrix{T}; tol::Real=1e-10) where T
        @assert size(matrix, 1) == size(matrix, 2) "Matrix must be square"
        # Check symmetry by comparing the matrix to its adjoint (conjugate transpose)
        return maximum(abs.(matrix .- adjoint(matrix))) < tol
    end
end #module end
