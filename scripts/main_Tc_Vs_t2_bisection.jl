#-----------------------------------
#   DESCRIPTION: This program calculates critical temperature Tc Vs μ
#   See paper PRM. 2, 024801 (2018) by A. Ptok et.al(Fig. 3)
#
using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
    using Printf
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

    # === CSV logging setup ===
    csvfile = "$(logFolder)Tc_calculation_log_$t2.csv"
    open(csvfile, "w") do io
        println(io, "iter,timestamp,T_high,T_low,T_try,converged,Δ̄,T_diff")
    end
    # =========================
    # Bisection parameters
    T_low = 0.0
    T_high = 0.5
    T_tol = 5e-3
    rel_tol = 5e-2
    max_iter = 30

    Δ₀ = nothing
    Tc = 0.0
    iter = 0

    while abs(T_high - T_low) > T_tol && iter < max_iter
        # Use T_low in first iteration to calculate Δ0
        T_try = (iter == 0) ? T_low : 0.5 * (T_low + T_high)

        n_up = zeros(Float64, nSites)
        n_dn = zeros(Float64, nSites)
        deltaOld = ones(Float64, nSites)

        deltaFinal, _, _, _, _, _, _, _, isConverged, _, count =
            run_self_consistency_numpy(deltaOld, μ, nSites, n_up, n_dn, tMat, U, J,
                                        impuritySite, T_try; tol=tol, maxCount=maxCount,
                                        isComplexCalc=isComplexCalc)

        if !isConverged
            write_log(logFileName, "WARNING", "iter=$iter, T = $T_try did not converge. Skipping.")
            T_high = T_try
            iter += 1
            continue
        end

        Δ̄ = abs(sum(deltaFinal) / nSites)

        if iter == 0
            Δ₀ = Δ̄
            if Δ₀ < 1e-6
                Tc = T_low  # or Tc = 0.0
                write_log(logFileName, "INFO", "Gap very small at T=$T_low. Storing Tc = $Tc")
                open(fileName, "a") do io
                    println(io, "$t2\t$Tc")
                end
                return
            end

            # First run just establishes Δ₀
            iter += 1
            continue
        end

        write_log(logFileName, "INFO", "iter=$iter, T=$T_try, Δ̄=$Δ̄")
        # CSV append with timestamp, Δ̄, and 3 decimal precision
        open(csvfile, "a") do io
            ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
            diff = T_high - T_low
            println(io,
                "$iter,$ts," *
                @sprintf("%.6f", T_high) * "," *
                @sprintf("%.6f", T_low) * "," *
                @sprintf("%.6f", T_try) * "," *
                (isConverged ? "yes" : "no") * "," *
                @sprintf("%.4f", Δ̄) * "," *
                @sprintf("%.6f", diff)
            )
        end

        if Δ̄ / Δ₀ < rel_tol
            T_high = T_try
        else
            T_low = T_try
        end

        iter += 1
    end

    Tc = round(0.5 * (T_low + T_high), digits=3)

    open(fileName, "a") do io
        println(io, "$t2\t$Tc")
    end
    et = time() - st
    message = string("Completed in: ", format_elapsed_time(et))
    write_log(logFileName, "INFO", message)
    GC.gc()
end

timestart = time()
@everywhere begin
    # tMatfileName = "$(dataSetFolder)ham_$(N)_t_$t"
    tMat = generate_t_matrix(tMatfileName)
    if !isreal(tMat)
        tMat = real(tMat)
    end
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
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))
