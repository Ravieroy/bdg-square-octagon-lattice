#-----------------------------------
#   DESCRIPTION: This program calculates critical temperature Tc Vs μ
#   for the spinfull case. λ ≡ Rashba SOC
#   λ_iso ≡ Intrinsic SOC

using Distributed
@everywhere begin
    using DelimitedFiles
    using LinearAlgebra
    using Dates
    using PyCall
    using Statistics
    using CSV
    using DataFrames
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

@everywhere function main(λ)
    st = time()
    logFileName = "$(logFolder)soc_lambda_$λ.log"
    fileName = "$(saveInFolder)soc_Tc_lambda_$λ.txt"
    message = string("Running for λ=$λ, id=$(myid())")
    write_log(logFileName, message)

    # === CSV logging setup ===
    csvfile = "$(logFolder)Tc_calculation_log_$λ.csv"
    open(csvfile, "w") do io
        println(io, "iter,timestamp,T_high,T_low,T_try,converged,Δ̄,T_diff")
    end
    # =========================

    # Bisection parameters
    T_low = 0.0
    T_high = 1.0
    T_tol = 1e-3
    rel_tol = 1e-2
    max_iter = 15

    Δ₀ = nothing
    Tc = 0.0
    iter = 0

    while abs(T_high - T_low) > T_tol && iter < max_iter
        # Use T_low in first iteration to calculate Δ0
        T_try = (iter == 0) ? T_low : 0.5 * (T_low + T_high)
        n_up = zeros(Float64, nSites)
        n_dn = zeros(Float64, nSites)
        deltaOld = ones(Float64, nSites)

        deltaFinal, _, _, _, _, _, isConverged, endTime, count =
            run_self_consistency_numpy_spinfull(deltaOld,
                μ, nSites, n_up, n_dn, tMat, U, J, x, y, neighbors, λ, λ_iso,
                t_map, iso_map, impuritySite, T_try, tol=tol, maxCount=maxCount,
                isComplexCalc=isComplexCalc)

        Δ̄ = mean(abs.(deltaFinal))

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

        if !isConverged
            write_log(logFileName, "WARNING", "iter=$iter, T = $T_try did not converge. Skipping.")
            T_high = T_try
            iter += 1
            continue
        end

        if iter == 0
            Δ₀ = Δ̄
            if Δ₀ < 1e-6
                Tc = T_low  # or Tc = 0.0
                write_log(logFileName, "INFO", "Gap very small at T=$T_low. Storing Tc = $Tc")
                open(fileName, "a") do io
                    println(io, "$λ\t$Tc")
                end
                return
            end

            # First run just establishes Δ₀
            iter += 1
            continue
        end

        write_log(logFileName, "INFO", "iter=$iter, T=$T_try, Δ̄=$Δ̄")

        if Δ̄ / Δ₀ < rel_tol
            T_high = T_try
        else
            T_low = T_try
        end

        iter += 1
    end

    Tc = round(0.5 * (T_low + T_high), digits=3)

    open(fileName, "a") do io
        println(io, "$λ\t$Tc")
    end

    et = time() - st
    message = string("Completed in: ", format_elapsed_time(et))
    write_log(logFileName, "INFO", message)
    GC.gc()
end # main end

timestart = time()
@everywhere begin
    # tMatfileName = "$(dataSetFolder)ham_$(N)_t_$t"
    tMat = generate_t_matrix(tMatfileName)
    df = CSV.read("$(dataSetFolder)df_square-octagon$N.csv", DataFrame)
    x = df.x
    y = df.y
    neighbors = hcat(df.n1, df.n2, df.n3, df.n4)
    t_map = generate_t_map(df, t2)
    # iso_map = compute_iso_map(df)
    iso_map = nothing
end

infoLogFileName = "$(logFolder)info.log"
pf = @__FILE__

# Detailed description of the program
desc = """
Calculates the critical temperature Tc for various chemical potential λ.
for the spinfull case.
The program takes μVals as input and processes each value in parallel.
For each μ, the program calculates and stores the following data:
    - λ : Chemical Potential
    - Tc : critical temperature
"""

if !isfile(infoLogFileName)
    message = string("Program name $pf")
    write_log(infoLogFileName, "INFO", message)
    write_log(infoLogFileName, desc)
end

column_names = ["λ", "Tc"]
tcFileNameFinal = "$(saveInFolder)Tc.txt"
# Open the file for writing
if !isfile(tcFileNameFinal)
    open(tcFileNameFinal, "w") do io
        # Write column names as the first row
        println(io, join(column_names, "\t"))
    end
end

pmap(main, λVals)
elapsed = time() - timestart
timeNow, dateToday = get_present_time()
println("($timeNow)Total Time Taken: ", format_elapsed_time(elapsed))

