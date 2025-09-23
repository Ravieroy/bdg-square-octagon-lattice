** The file structure and codes are a bit rough around the edges right now. More details and files along with pedagogy notebooks will be added soon **

This code uses the Bogoliubov-de-Gennes method for the spin full case to study the superconductivity for a very special type of lattice known as Square-Octagon lattice. The basics of the BdG method can be looked from the seminal paper by A.Ghosal et al. (PHYSICAL REVIEW B, VOLUME 65, 014501). The spinfull case can be studied by another beautiful set of studies by Andrzej Ptok (for e.g. in PHYSICAL REVIEW B 96, 184425 (2017))

The basic structure of the directory for running any BdG calculation is :
```
├── `data`
├── `logs`
├── `results`
├── `scripts`
└── `src`
```

[!NOTE]

Make sure these subdirectories are present in your current working directory, and you execute the codes from `scripts` directory.

`data`: This directory is used to store all the important input files needed to run the BdG calculation. For example `df_square-octagon10.csv`.

`logs`: This directory is used to store log files.

`results`: This directory is used to store results from the BdG calculation.

`scripts`: This is the directory where all the calculation is run. It contains all the script files needed to run any BdG calculation. One doesn't need to use these scripts as new scripts can be written as per requirement of the calculation. The core code is in the src file and these scripts can serve as examples for how to do that. If SOC, defined by the value of λ is zero, it is better to use the scripts starting with `main_...jl` and use `soc_...jl` versions for the SOC calculations. Although they would both give the same result, except the former is faster.

```
├── check_julia_dependency.jl
├── check_python_dependency.py
├── main_delta_U_mu_3D.jl
├── main_delta_Vs_mu.jl
├── main_delta_Vs_t2.jl
├── main_delta_Vs_T_varying_mu.jl
├── main_delta_Vs_U_varying_mu.jl
├── main_delta_Vs_U_varying_t2.jl
├── main_Tc_Vs_mu_bisection.jl
├── main_Tc_Vs_t2_bisection.jl
├── main_Tc_Vs_t2.jl
├── Makefile
├── makegrid.py
├── params.jl
├── soc_delta_Vs_lambda_varying_t2.jl
├── soc_delta_Vs_mu.jl
├── soc_delta_Vs_T_varying_lambda.jl
├── soc_delta_Vs_U_varying_lambda.jl
├── soc_delta_Vs_U_varying_mu.jl
├── soc_delta_Vs_U_varying_t2.jl
├── soc_Tc_Vs_lambda_bisection.jl
├── soc_Tc_Vs_lambda.jl
├── soc_Tc_Vs_mu_bisection.jl
└── soc_Tc_Vs_t2_bisection.jl
```

`src`: This directory contains all the source files needed to run a BdG calculation.

```
├── `bdg_utilities.jl` : All the utilities required for running BdG calculation
├── `external_utils.jl` : Some functions used for specific case, when Hamiltonian is taken from external source(e.g. FORTRAN). Usually not required for general calculation.
├── `generalutils.jl`: Contains all the general utilities which is useful for BdG calculation or other places.
└── `logging_utils.jl`: Utilities required for logging.
```

## Workflow 

1. **Step 0 :** Create the necessary directories. data, logs, results alongwith src and scripts
1. **Step 1 :** Run Makefile for fresh calculation. `make fresh`
2. **Step 2 :** Edit the `params.jl` file accordingly
4. **Step 3 :** Run `julia -p <n> script_name.jl` on `n` cores using the script `script_name.jl`

