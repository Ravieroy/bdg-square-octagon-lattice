#-----------------------------------
#   This program calculates the order parameter on a grid of U and mu.
#   See paper PRA. 97, 053619 (2018) by Agnieszka Cichy et.al(Fig. 3)
#
using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
    using CSV, DataFrames
    @everywhere(@pyimport numpy as np)
    # Importing personal modules
    include("../src/bdg_utilities.jl")
    include("../src/general_utils.jl")
    include("../src/logging_utils.jl")
    include("params.jl")
end # module everywhere end

# Loading personal modules
@everywhere using .BdGUtilities
@everywhere using .GeneralUtils
@everywhere using .LoggingUtils

function main()
    logFileName = "$(logFolder)status.log"
    write_log(logFileName, "Calculation for Δ on U-μ grid started")

    numUs = length(UList)
    numMus = length(μVals)

    mus = repeat(μVals, 1, numUs)
    Us = repeat(UList', numMus, 1)

    avgDelta = zeros(Float64, numMus, numUs)

    total_iterations = numMus * numUs
    counter = 0

    tMat = generate_t_matrix(tMatfileName)
    if !isreal(tMat)
        tMat = real(tMat)
    end
    for i in 1:numMus
        for j in 1:numUs
            μ = mus[i, j]
            U = Us[i, j]

            deltaFinal, _, _, _, _, _, _, _, isConverged, endTime, count =
                run_self_consistency_numpy(
                    deltaOld,
                    μ,
                    nSites,
                    n_up,
                    n_dn,
                    tMat,
                    U,
                    J,
                    impuritySite,
                    T,
                    tol=tol,
                    maxCount=maxCount,
                    isComplexCalc=isComplexCalc
                )
            if isConverged == true
                message = string("Converged in $count iterations")
                write_log(logFileName, "SUCCESS", message)
            else
                message = string("Calculation did not converge")
                write_log(logFileName, "WARNING", message)
            end

            Δ̄ = sum(deltaFinal) / nSites
            if !isreal(Δ̄)
                Δ̄ = abs(Δ̄)
            end

            avgDelta[i, j] = Δ̄

            counter += 1
            message = string("Completed: $counter / $total_iterations (", round(counter / total_iterations * 100, digits=2), "%)")
            write_log(logFileName, message)
        end
    end

    # Create header names based on U values
    col_headers = ["U$(j)" for j in 1:numUs]

    # Create DataFrame with custom column names
    df = DataFrame(avgDelta, Symbol.(col_headers))

    # Convert the matrix to a DataFrame
    # df = DataFrame(avgDelta, :auto)

    # Save the DataFrame to a CSV file
    CSV.write("$(saveInFolder)avgDelta.csv", df)
    message = string("Matrix avgDelta saved as avgDelta.csv")
    write_log(logFileName, message)

    save_file(avgDelta, "$(saveInFolder)avgDelta.pkl")
    message = string("Matrix avgDelta saved as avgDelta.pkl")
    write_log(logFileName, message)
end  # main end

timestart = time()

main()
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

