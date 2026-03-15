using DelimitedFiles
using DataFrames
using DataStructures # for SortedDict
using CSV
ENV["GKSwstype"] = "100"  # Use offscreen rendering for headless environments
using Plots

include("../src/bdg_utilities.jl")
using .BdGUtilities

include("../src/general_utils.jl")
using .GeneralUtils

include("../src/external_utils.jl")
using .ExternalUtils

include("../scripts/params.jl")

folder_names = ["data", "logs", "results", "scripts", "src",]
base_path = joinpath(pwd(), "..")
create_folders_if_not_exist(folder_names, base_path)

# folder path and the list of filenames to check
folder_path = dataSetFolder
filenames = ["COORDA", "COORDB", "COORDC", "NN_MAP"]

# Check if each file exists in the folder
for filename in filenames
    file_path = joinpath(folder_path, filename)
    if !isfile(file_path)
        error("File does not exist: $filename")
    end
end

println("All files exist. Continuing with the script...")

NN_MAP = readdlm("$(dataSetFolder)NN_MAP")

# Get number of rows in NN_MAP
num_rows = size(NN_MAP, 1)

# Assert that nSites matches the number of rows
@assert nSites == num_rows "Mismatch in nSites: Expected $nSites sites, but NN_MAP has $num_rows rows. Please check your lattice parameters or NN_MAP file."

coordA = "$(dataSetFolder)COORDA"
coordB = "$(dataSetFolder)COORDB"
coordC = "$(dataSetFolder)COORDC"

create_df_square_octagon(nnMapFileName,
    coordA,
    coordB,
    coordC;
    saveLocally=saveDataFrame,
    fileName=dfName)

data = readdlm(dfName, ',')
center_site = find_center_site(dfName)
println("The site at the center is: ", center_site)
# Save center site information as-is
# center_site_output_file = joinpath(base_path, "results", "center_site_info.txt")
center_site_output_file = "$(dataSetFolder)center_site_info_$N.txt"
open(center_site_output_file, "w") do io
    println(io, string(center_site))
end
println("Center site information saved to $center_site_output_file")


# Visualization
file_path = "../results/lattice_with_impurity.png"

x_coords = data[2:end, 2]
y_coords = data[2:end, 3]
scatter(x_coords, y_coords, label="Sites", title="Lattice Sites", xlabel="X", ylabel="Y")
scatter!([center_site[2]], [center_site[3]], color=:red, label="Center Site", markersize=5)
savefig(file_path)

println("DONE!!")
