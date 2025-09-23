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

    """
        create_df_square_octagon(nnMapFileName, coordA, coordB, coordC; saveLocally::Bool=true, fileName::Union{Nothing,String}=nothing)

    Creates a DataFrame representing a square-octagon lattice with site coordinates, nearest-neighbor information, and sublattice labels.
    The files needed for this comes from a separate FORTRAN program which creates the Square-Octagon Lattice.

    # Arguments
    - `nnMapFileName::AbstractString`: Path to the text file containing raw neighbor map data (site positions and neighbors).
    - `coordA, coordB, coordC::AbstractString`: Paths to text files containing the x/y coordinates of sites belonging to sublattices A, B, and C, respectively.
    - `saveLocally::Bool=true`: If `true`, saves the resulting DataFrame to a CSV file specified by `fileName`.
    - `fileName::Union{Nothing,String}=nothing`: Optional output file name for saving the DataFrame as CSV; if `nothing`, the DataFrame is not saved.

    # Returns
    - `new_df::DataFrame`: A DataFrame with columns:
        - `:siteIndex`: Integer site index.
        - `:x, :y`: Real-space coordinates.
        - `:n1, n2, ..., nN`: Indices of nearest neighbors (with `missing` if neighbor is absent).
        - `:Sublattice`: Sublattice label ("A", "B", "C", or "D" if not matched to provided coordinates).

    # Notes
    - Detects and warns about duplicate neighbors in each site's neighbor list.
    - Checks for corrupted entries where neighbor indices might contain string `"missing"`.
    - Assigns sublattice labels by matching site coordinates to the lists in `coordA`, `coordB`, and `coordC` within a small tolerance (`ϵ=1e-6`).
    """

    function create_df_square_octagon(nnMapFileName, coordA, coordB, coordC;
                                      saveLocally::Bool=true, fileName::Union{Nothing,String}=nothing)

        # --- read raw neighbor map ---
        data = readdlm(nnMapFileName)
        df   = DataFrame(data, :auto)

        # how many neighbor pairs?
        n_neighbors = Int((ncol(df) - 2) ÷ 2)

        # build site→index map
        site_index_map = Dict{Tuple{Float64,Float64},Int}()
        for i in 1:nrow(df)
            site_index_map[(df[i,1], df[i,2])] = i
        end

        # --- prepare new_df with correct types ---
        new_df = DataFrame(
            siteIndex = Int[],
            x = Float64[],
            y = Float64[]
            )
        for k in 1:n_neighbors
            new_df[!, Symbol("n$k")] = Union{Int,Missing}[]
        end

        # --- populate it ---
        for i in 1:nrow(df)
            idx = site_index_map[(df[i,1], df[i,2])]

            neighs = Union{Int,Missing}[]
            for j in 3:2:(2 + 2*n_neighbors)
                pos = (df[i, j], df[i, j+1])
                push!(neighs, get(site_index_map, pos, missing))
            end

            # === Duplicate neighbor check ===
            # Remove `missing` before checking
            filtered_neighs = filter(!ismissing, neighs)
            dupes = findall(x -> count(==(x), filtered_neighs) > 1, filtered_neighs)
            if !isempty(dupes)
                println("WARNING: Duplicate neighbor(s) found at site index $idx → ", filtered_neighs[dupes])
            end

            row = Dict{Symbol,Any}()
            row[:siteIndex] = idx
            row[:x] = Float64(df[i, 1])
            row[:y] = Float64(df[i, 2])
            for k in 1:n_neighbors
                row[Symbol("n$k")] = neighs[k]
            end

            push!(new_df, row)
        end


            # check for string "missing"
            if any(col -> any(x -> x == "missing", col), eachcol(new_df))
                @warn "WARNING: Corrupted Dataframe"
            end

            # --- tag sublattices ---
            coorda = readdlm(coordA); coorda_df = DataFrame(coorda, [:x, :y])
            coordb = readdlm(coordB); coordb_df = DataFrame(coordb, [:x, :y])
            coordc = readdlm(coordC); coordc_df = DataFrame(coordc, [:x, :y])

            ϵ = 1e-6
            is_match(x1,y1,x2,y2) = abs(x1-x2)<ϵ && abs(y1-y2)<ϵ
            in_coords(x,y,df2)    = any(r -> is_match(r[:x],r[:y],x,y), eachrow(df2))

            new_df.Sublattice = Vector{String}(undef, nrow(new_df))
            for i in 1:nrow(new_df)
                x0, y0 = new_df[i, :x], new_df[i, :y]
                if      in_coords(x0,y0, coorda_df) new_df[i, :Sublattice] = "A"
                    elseif  in_coords(x0,y0, coordb_df) new_df[i, :Sublattice] = "B"
                    elseif  in_coords(x0,y0, coordc_df) new_df[i, :Sublattice] = "C"
                    else                                 new_df[i, :Sublattice] = "D"
                end
            end

            # --- optional save ---
            if saveLocally
                if fileName !== nothing
                    CSV.write(fileName, new_df)
                else
                    @info "File name not provided. Skipping save."
                end
            end

            return new_df
        end



    """
        find_center_site(dfName::AbstractString)

    Finds the site closest to the geometric center of a lattice from a CSV-like data file containing site coordinates.

    # Arguments
    - `dfName::AbstractString`: Path to the CSV file with a header row followed by rows of site data; assumes x- and y-coordinates are in the second and third columns, respectively.

    # Returns
    - `row::Vector`: The row of data (as a vector) corresponding to the site closest to the center of the lattice, including all columns from the input file.

    # Notes
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
