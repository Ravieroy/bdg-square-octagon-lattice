module ExternalUtils
    using DelimitedFiles
    using DataFrames
    using DataStructures # for SortedDict
    using CSV
    using Plots
    include("../scripts/params.jl")

    export create_df_square_octagon
    export find_center_site

    function create_df_square_octagon(nnMapFileName, coordA, coordB, coordC; saveLocally=true, fileName=nothing)
        data = readdlm(nnMapFileName)
        df = DataFrame(data, :auto)

        # map site position => site index
        site_index_map = Dict{Tuple{Float64,Float64},Int}()
        for i in 1:size(df, 1)
            site_index_map[(df[i, 1], df[i, 2])] = i
        end
        new_df = DataFrame(
            siteIndex=Int[],
            x=Float64[],
            y=Float64[],
            n1=Union{Int,Missing}[],
            n2=Union{Int,Missing}[],
            n3=Union{Int,Missing}[],
            n4=Union{Int,Missing}[])

        # Populate the new DataFrame
        for i in 1:size(df, 1)
            site_index = site_index_map[(df[i, 1], df[i, 2])]
            neighbors = Union{Int,Missing}[]

            for j in 3:2:9
                neighbor_pos = (df[i, j], df[i, j+1])
                neighbor_index = get(site_index_map, neighbor_pos, missing)
                push!(neighbors, neighbor_index)
            end

            push!(new_df, vcat([site_index, df[i, 1], df[i, 2]], neighbors))
        end

        #check for any wrong/missing data in the dataframe
        has_string_missing = any(col -> any(x -> x == "missing", col), eachcol(new_df))

        if has_string_missing ≠ false
            println("WARNING: Corrupted Dataframe")
        end

        # Load the coordinate files
        coorda = readdlm(coordA)
        coordb = readdlm(coordB)
        coordc = readdlm(coordC)

        # Convert the coordinate arrays to DataFrames
        coorda_df = DataFrame(coorda, [:x, :y])
        coordb_df = DataFrame(coordb, [:x, :y])
        coordc_df = DataFrame(coordc, [:x, :y])

        # Add sublattice information to the main DataFrame
        new_df.Sublattice = Vector{String}(undef, nrow(new_df))

        # Iterate through each row of the main DataFrame
        for i in 1:nrow(new_df)
            x_pos = new_df[i, :x]
            y_pos = new_df[i, :y]

            if any(row -> row[:x] == x_pos && row[:y] == y_pos, eachrow(coorda_df))
                new_df[i, :Sublattice] = "A"
            elseif any(row -> row[:x] == x_pos && row[:y] == y_pos, eachrow(coordb_df))
                new_df[i, :Sublattice] = "B"
            elseif any(row -> row[:x] == x_pos && row[:y] == y_pos, eachrow(coordc_df))
                new_df[i, :Sublattice] = "C"
            else
                new_df[i, :Sublattice] = "D"
            end
        end

        if saveLocally == true && fileName ≠ nothing
            CSV.write(fileName, new_df)
        elseif saveLocally == true && fileName == nothing
            CSV.write("$(dataSetFolder)df_so_$N.csv", new_df)
        else
            println("Logic error in source code")
        end

        return new_df
    end


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
