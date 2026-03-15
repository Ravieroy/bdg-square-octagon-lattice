module ExternalUtils
    using DelimitedFiles
    using DataFrames
    using DataStructures # for SortedDict
    using CSV
    using Plots
    include("../scripts/params.jl")

    export create_df_square_octagon
    export find_center_site
    export plot_lattice

    function create_df_square_octagon(nnMapFileName::AbstractString,
                                  coordA::AbstractString,
                                  coordB::AbstractString,
                                  coordC::AbstractString;
                                  saveLocally::Bool=true,
                                  fileName::Union{Nothing,String}=nothing,
                                  return_raw::Bool=false,
                                  saveRaw::Bool=true,
                                  rawFileName::Union{Nothing,String}=nothing)

        # --- read neighbor map ---
        raw = readdlm(nnMapFileName)
        df_raw = DataFrame(raw, :auto)

        n_neighbors_raw = Int((ncol(df_raw) - 2) ÷ 2)
        if n_neighbors_raw < 4
            error("Expected at least 4 neighbor pairs per row (3 NN + 1 NNN). Found $n_neighbors_raw.")
        end

        # build site -> index
        site_index_map = Dict{Tuple{Float64,Float64}, Int}()
        for i in 1:nrow(df_raw)
            site_index_map[(Float64(df_raw[i,1]), Float64(df_raw[i,2]))] = i
        end

        function neighbor_index_from_row(i::Int, k::Int)
            colx = 2 + 2*(k-1) + 1
            coly = colx + 1
            pos = (Float64(df_raw[i, colx]), Float64(df_raw[i, coly]))
            return get(site_index_map, pos, missing)
        end

        # ---------- RAW NEIGHBOR DATAFRAME r1..r8 ----------
        raw_idx_df = DataFrame(siteIndex = Int[], x = Float64[], y = Float64[],
                               r1 = Union{Int,Missing}[], r2 = Union{Int,Missing}[],
                               r3 = Union{Int,Missing}[], r4 = Union{Int,Missing}[],
                               r5 = Union{Int,Missing}[], r6 = Union{Int,Missing}[],
                               r7 = Union{Int,Missing}[], r8 = Union{Int,Missing}[])

        nsites = nrow(df_raw)
        for i in 1:nsites
            x0 = Float64(df_raw[i,1]); y0 = Float64(df_raw[i,2])
            r = [k <= n_neighbors_raw ? neighbor_index_from_row(i, k) : missing for k in 1:8]
            push!(raw_idx_df, (i, x0, y0, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
        end

        # save raw dataframe
        if saveRaw
            if rawFileName !== nothing
                CSV.write(rawFileName, raw_idx_df)
            else
                @info "Raw file name not provided. Skipping raw save."
            end
        end

        # ---------- SUBLATTICE ASSIGNMENT ----------
        coorda = readdlm(coordA); coorda_df = DataFrame(coorda, [:x, :y])
        coordb = readdlm(coordB); coordb_df = DataFrame(coordb, [:x, :y])
        coordc = readdlm(coordC); coordc_df = DataFrame(coordc, [:x, :y])

        epsilon = 1e-6
        is_match(x1,y1,x2,y2) = abs(x1-x2) < epsilon && abs(y1-y2) < epsilon
        in_coords(x,y,df2) = any(r -> is_match(r[:x], r[:y], x, y), eachrow(df2))

        sublattice = Vector{String}(undef, nsites)
        for i in 1:nsites
            x0 = Float64(df_raw[i,1]); y0 = Float64(df_raw[i,2])
            if in_coords(x0,y0, coorda_df)
                sublattice[i] = "A"
            elseif in_coords(x0,y0, coordb_df)
                sublattice[i] = "B"
            elseif in_coords(x0,y0, coordc_df)
                sublattice[i] = "C"
            else
                sublattice[i] = "D"
            end
        end

        # ---------- ORDERED OUTPUT ----------
        out = DataFrame(siteIndex = Int[], x = Float64[], y = Float64[],
                        n1 = Union{Int,Missing}[], n2 = Union{Int,Missing}[],
                        n3 = Union{Int,Missing}[], n4 = Union{Int,Missing}[],
                        n5 = Union{Int,Missing}[], n6 = Union{Int,Missing}[],
                        n7 = Union{Int,Missing}[], n8 = Union{Int,Missing}[],
                        Sublattice = String[])

        for i in 1:nsites
            x0 = Float64(df_raw[i,1]); y0 = Float64(df_raw[i,2])
            raw_neighs = [k <= n_neighbors_raw ? neighbor_index_from_row(i, k) : missing for k in 1:8]

            function find_nn_by_sublabel(lbl)
                for s in 1:3
                    nb = raw_neighs[s]
                    if !ismissing(nb) && sublattice[nb] == lbl
                        return nb
                    end
                end
                return missing
            end

            nnn_idx = raw_neighs[4]
            site_label = sublattice[i]

            n1 = missing; n2 = missing; n3 = missing; n4 = missing
            if site_label == "A"
                n1 = find_nn_by_sublabel("D")
                n2 = find_nn_by_sublabel("B")
                n3 = find_nn_by_sublabel("C")
                n4 = nnn_idx
            elseif site_label == "B"
                n1 = nnn_idx
                n2 = find_nn_by_sublabel("D")
                n3 = find_nn_by_sublabel("A")
                n4 = find_nn_by_sublabel("C")
            elseif site_label == "C"
                n1 = find_nn_by_sublabel("D")
                n2 = find_nn_by_sublabel("B")
                n3 = nnn_idx
                n4 = find_nn_by_sublabel("A")
            else
                n1 = find_nn_by_sublabel("B")
                n2 = nnn_idx
                n3 = find_nn_by_sublabel("A")
                n4 = find_nn_by_sublabel("C")
            end

            push!(out, (i, x0, y0,
                        n1, n2, n3, n4,
                        raw_neighs[5], raw_neighs[6], raw_neighs[7], raw_neighs[8],
                        site_label))
        end

        if saveLocally
            if fileName !== nothing
                CSV.write(fileName, out)
            else
                @info "File name not provided. Skipping ordered save."
            end
        end

        return return_raw ? (out, raw_idx_df) : out
    end



    """
        find_center_site(dfName::AbstractString)

    Finds the site closest to the geometric center of a lattice from a CSV-like data file containing site coordinates.

    Arguments
    - `dfName::AbstractString`: Path to the CSV file with a header row followed by rows of site data; assumes x- and y-coordinates are in the second and third columns, respectively.

    Returns
    - `row::Vector`: The row of data (as a vector) corresponding to the site closest to the center of the lattice, including all columns from the input file.

    Notes
    - Computes the geometric center as the midpoint of the min and max x/y coordinate ranges.
    - Uses squared Euclidean distance to determine proximity to the center.
    - Assumes the file has a header; indexing accounts for the header by offsetting the returned row index.
    """

    function find_center_site(dfName)
        data = readdlm(dfName, ',')
        # Extract x and y coordinates
        x_coords = data[2:end, 2]
        y_coords = data[2:end, 3]

        # Center of the lattice (average of min and max values)
        center_x = (minimum(x_coords) + maximum(x_coords)) / 2
        center_y = (minimum(y_coords) + maximum(y_coords)) / 2

        # Calculate the distance of each point from the center
        distances = [(x - center_x)^2 + (y - center_y)^2 for (x, y) in zip(x_coords, y_coords)]

        # Find the index of the minimum distance
        min_index = argmin(distances) + 1  # +1 because data is offset by the header row

        # Return the row of data at the minimum distance
        return data[min_index, :]
    end


    """
        plot_lattice(file_path::String, output_dir::String)

    Loads lattice data from `file_path`, finds the center site,
    plots the lattice sites with the center highlighted, and saves the plot in
    `output_dir`.
    """
    function plot_lattice(file_path::String, output_dir::String)
        data = readdlm(file_path, ',')

        center_site = find_center_site(data)
        println("The site at the center is: ", center_site)

        # Extract x and y coordinates for visualization
        x_coords = data[2:end, 2]
        y_coords = data[2:end, 3]

        # Create scatter plot
        scatter(x_coords, y_coords, label="Sites", title="Lattice Sites", xlabel="X", ylabel="Y")
        scatter!([center_site[2]], [center_site[3]], color=:red, label="Center Site", markersize=5)

        # Save plot to specified folder
        output_path = joinpath(output_dir, "lattice_plot.png")
        savefig(output_path)
        println("Plot saved to: ", output_path)
    end
end # module end
