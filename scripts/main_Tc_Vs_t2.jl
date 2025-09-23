#-----------------------------------
#   DESCRIPTION: This program calculates critical temperature Tc Vs t₂

using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
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

@everywhere function main(t2)
    st = time()
    logFileName = "$(logFolder)t2_$t2.log"
    fileName = "$(saveInFolder)Tc_t2_$t2.txt"
    message = string("Running for t2=$t2, id=$(myid())")
    write_log(logFileName, message)
    tMatfileName = "$(dataSetFolder)ham_$(N)_t2_$t2"
    tMat = generate_t_matrix(tMatfileName)
    if !isreal(tMat)
        tMat = real(tMat)
    end

    dT = 0.1
    # T = 0.0
    T = T_start_map[λ]
    foundTc = false
    iter = 1
    while foundTc == false
        message = string("Running for T=$T")
        write_log(logFileName, "INFO", message)
        n_up = 0 * ones(Float64, nSites)
        n_dn = 0 * ones(Float64, nSites)
        deltaOld = ones(Float64, nSites)
        deltaFinal, _, _, _, _, _, _, _, isConverged, _, count =
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
            # Δ̄ = sum(deltaFinal) / nSites
        end

        Δ̄ = sum(deltaFinal) / nSites
        if !isreal(Δ̄)
            Δ̄ = abs(Δ̄)
        end

        message = string("iter=$iter, Δ̄=$Δ̄")
        write_log(logFileName, message)

        if Δ̄ < 0.01
            foundTc = true
            Δ̄ = round(Δ̄, digits=3)
            Tc = round(T, digits=3)
            list = [t2, Tc]
            open(fileName, "a") do io
                println(io, join(list, '\t'))
            end
        end
        iter += 1
        T = T + dT
    end # while(foundTc) loop end

    et = round(time() - st, digits=2)
    column_names = ["t2", "Tc"]
    tcFileNameFinal = "$(saveInFolder)Tc.txt"
    # Open the file for writing
    if !isfile(tcFileNameFinal)
        open(tcFileNameFinal, "w") do io
            # Write column names as the first row
            println(io, join(column_names, "\t"))
        end
    end
    message = string("Completed in $et seconds")
    write_log(logFileName, "INFO", message)
    GC.gc()
end  # main end

timestart = time()
@everywhere begin
    T_start_map = Dict(
    0.0 => 0.1,
    0.5 => 0.25,
    1.0 => 0.25,
    )
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the critical temperature Tc for various chemical potential t'.
The program takes t2Vals as input and processes each value in parallel.
For each μ, the program calculates and stores the following data:
    - t2 : NNN hopping parameter
    - Tc : critical temperature
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

pmap(main, t2Vals)
elapsed = round(time() - timestart, digits=2)
timeNow, dateToday = get_present_time()
println("($timeNow)The elapsed time : $elapsed secs ($(round(elapsed/60, digits = 2)) mins)")

