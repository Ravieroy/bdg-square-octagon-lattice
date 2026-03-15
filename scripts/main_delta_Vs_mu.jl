#-----------------------------------
#   DESCRIPTION: This program calculates the order parameter(Δ) for various
#   values of chemical potential mu for non-spinfull case.
#
#   It runs the program for given values of mu in parallel and stores the
#   results in a text file.
#
#   See paper PRB. 105, 214510 (2022) by Agnieszka Cichy et.al(Fig. 2c)

using Distributed

@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Statistics
    using Dates
    using CSV
    using DataFrames
    using PyCall
    using Statistics
    @everywhere(@pyimport numpy as np)

    include("../src/bdg_utilities.jl")
    include("../src/general_utils.jl")
    include("../src/logging_utils.jl")
    include("params.jl")
end

@everywhere using .BdGUtilities
@everywhere using .GeneralUtils
@everywhere using .LoggingUtils

@everywhere begin
    function _init_shared(rawdfName::String, tMatFileName::String, t2::Float64)
        df = CSV.read(rawdfName, DataFrame)
        x, y, neighbors = extract_geometry(df)
        tMap = generate_t_map(neighbors; t1=1.0, t2=t2)
        # tMat = generate_t_matrix(tMatFileName)
        tMat = build_t_matrix(neighbors, tMap)
        return x, y, neighbors, tMap, tMat
    end

    const x, y, neighbors, tMap, tMat = _init_shared(rawdfName, tMatFileName, t2)
end

@everywhere function main(mu)
    logFile = "$(logFolder)mu_$(mu)_pid_$(myid()).log"
    outFile = "$(saveInFolder)mu_$(mu).txt"

    st = time()
    write_log(logFile, "INFO", "Running mu=$mu on worker $(myid())")

    # initial guesses
    nUp = copy(nUp0)
    nDn = copy(nDn0)
    deltaOld = ones(Float64, nSites)

    deltaFinal, _, _, nAvg, _, _, _, _, isConverged, endTime, count =
        run_self_consistency_numpy(
            deltaOld,
            mu, nSites,
            nUp, nDn,
            tMat, U, J,
            impuritySite, T;
            tol=tol, maxCount=maxCount,
            includeHartree=includeHartree,
            verboseLogIn=logFile
        )

    if isConverged
        write_log(logFile, "SUCCESS", "Converged in $count iterations; $endTime s")
    else
        write_log(logFile, "WARNING", "Did not converge in $count iterations; $endTime s")
    end

    deltaAvg = mean(abs.(deltaFinal))

    open(outFile, "a") do io
        println(io, join([mu, deltaAvg], '\t'))
    end

    et = round(time() - st, digits=2)
    write_log(logFile, "INFO", "Completed mu=$mu in $(format_elapsed_time(et))")

    # cleanup
    evecs = nothing
    evals = nothing
    deltaFinal = nothing
    deltaOld = nothing
    nUp = nothing
    nDn = nothing
    nAvg = nothing

    pyimport("gc")[:collect]()
    GC.gc()

    return nothing
end

timestart = time()

desc = """
    Calculates deltaAvg vs mu (cannot use λ).
For each mu, stores:
    - mu
    - deltaAvg = mean(|Δ_i|)
"""

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__
if !isfile(infoLogFileName)
    write_log(infoLogFileName, "INFO", "Program name $pf")
    write_log(infoLogFileName, desc)
end


fileName = "$(saveInFolder)mu.txt"
if !isfile(fileName)
    list = ["mu", "ΔAvg"]
    open(fileName, "a") do io
        println(io, join(list, '\t'))
    end
end

pmap(main, muVals)

elapsed = round(time() - timestart, digits=2)
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

