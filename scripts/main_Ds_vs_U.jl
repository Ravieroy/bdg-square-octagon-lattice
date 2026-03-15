using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
    using Statistics
    using CSV
    using DataFrames
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

@everywhere function main(U)
    logFile = "$(logFolder)main_stiffness_U_$U.log"
    fileName = "$(saveInFolder)main_stiffness_U_$U.txt"
    message = string("Running for U=$U, id=$(myid())")
    write_log(logFile, message)
    st = time()

    # initial guesses
    nUp = copy(nUp0)
    nDn = copy(nDn0)
    deltaOld = copy(deltaOld0)

    deltaFinal, _, _, nAvg, evecs, evals, isConverged, endTime, count =
            run_self_consistency_numpy_spinfull_twisted(
                deltaOld,
                mu,
                nSites,
                nUp,
                nDn,
                U,
                J,
                x,
                y,
                neighbors,
                lambda,
                lambdaIso,
                tMap,
                isoMap,
                impuritySite,
                T,
                twist = twist,
                K = K,
                tol = tol,
                maxCount=maxCount,
                verboseLogIn = logFile
            )
        if isConverged == true
            message = string("Converged in $count iterations in $endTime s")
            write_log(logFile, "SUCCESS", message)
        else
            message = string("Calculation did not converge in $endTime s")
            write_log(logFile, "WARNING", message)
        end

        Ds = compute_Ds_helicity( deltaFinal, mu, nSites, nUp, nDn, U,
                impuritySite, J, x, y, neighbors, lambda, tMap, T;
                K=K, twist=twist)

        # Replace NaN / Inf by zero
        if !isfinite(Ds)
            Ds = 0.0
        end

        et = round(time() - st, digits=2)

        list = [U, Ds]
        open(fileName, "a") do io
            println(io, join(list, '\t'))
        end

        column_names = ["U", "Ds"]
        tcFileNameFinal = "$(saveInFolder)stiffness_vs_U.txt"
        # Open the file for writing
        if !isfile(tcFileNameFinal)
            open(tcFileNameFinal, "w") do io
                # Write column names as the first row
                println(io, join(column_names, "\t"))
            end
        end
        message = string("Completed for U=$U in ", format_elapsed_time(et))
        write_log(logFile, message)
        GC.gc()
end  # main end

timestart = time()
@everywhere begin
    df = CSV.read(rawdfName, DataFrame)
    x, y, neighbors = extract_geometry(df)
    tMap = generate_t_map(neighbors, t2=t2)
    # isoMap = compute_iso_map(df)
    isoMap = nothing
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
    Calculates the stiffness (Dₛ)
The program takes UVals as input and processes each value in parallel.
For each values of U, the program calculates and stores the following data:
    U: Interaction strength
    Ds: Stiffness
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

pmap(main, UVals)
elapsed = round(time() - timestart, digits=2)
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))
