[![DOI](https://img.shields.io/badge/DOI-10.1103%2Ff7rt--5vwb-blue)](https://doi.org/10.1103/f7rt-5vwb)

Code accompanying the paper: [Phys. Rev. B **113**, 104512 (2026)](https://doi.org/10.1103/f7rt-5vwb)

This code uses the Bogoliubov-de-Gennes method for the spin full case to study the superconductivity for a very special type of lattice known as Square-Octagon lattice. The basics of the BdG method can be looked from the seminal paper by A.Ghosal et al. (PHYSICAL REVIEW B, VOLUME 65, 014501). The spinfull case can be studied by another beautiful set of studies by Andrzej Ptok (for e.g. in PHYSICAL REVIEW B 96, 184425 (2017))

A mildly pedagogical set of notebooks are added in the `notebooks/tutorial` directory where I have tried to explain and contain certain parts of the programs for easy understanding.

The basic structure of the directory for running any BdG calculation is :
```
├── `data`
├── `logs`
├── `results`
├── `scripts`
└── `src`
```

> [!IMPORTANT]
> Make sure these subdirectories are present in your current working directory, and you execute the codes from `scripts` directory.

`data`: This directory is used to store all the important input files needed to run the BdG calculation. For example `df_square-octagon10.csv`.

`logs`: This directory is used to store log files.

`results`: This directory is used to store results from the BdG calculation.

`scripts`: This is the directory where all the calculation is run. It contains all the script files needed to run any BdG calculation. One doesn't need to use these scripts as new scripts can be written as per requirement of the calculation. Some example scripts to calculate some plots from the paper [Effect of Rashba spin-orbit coupling and next-nearest-neighbor hopping on superconductivity in the square-octagon lattice](https://doi.org/10.1103/f7rt-5vwb) are given here. The core code is in the `src` file and these scripts can serve as examples for how to do that. If SOC, defined by the value of λ is zero, it is better to use the scripts starting with `main_...jl` and use `soc_...jl` versions for the SOC calculations. Although they would both give the same result, except the former is faster.

```
├── check_julia_dependency.jl : checks and installs julia packages needed for the project
├── main_delta_Vs_mu.jl
├── main_delta_Vs_U_varying_t2.jl
├── main_Ds_vs_N.jl
├── main_Ds_vs_T.jl
├── main_Ds_vs_U.jl
├── Makefile : Makefile for quick actions (`reset` will not touch the data folder, `fresh` will empty all folders)
├── params.jl : Parameter file
├── plot.gp : A basic and quick gnuplot plotting script
├── soc_delta_Vs_mu.jl
├── soc_delta_Vs_T_varying_lambda.jl
└── soc_triplet_vs_RSOC.jl
```

`src`: This directory contains all the source files needed to run a BdG calculation.

```
├── `bdg_utilities.jl` : All the utilities required for running BdG calculation
├── `external_utils.jl` : Some functions used for specific case, when Hamiltonian is taken from external source(e.g. FORTRAN). Usually not required for general calculation.
├── `generalutils.jl`: Contains all the general utilities which is useful for BdG calculation or other places.
└── `logging_utils.jl`: Utilities required for logging.
```

## Files needed for running any calculation

In the current form following files are needed to run the calculation (say you are running for lattice size N=5, t2=1.0) :

1. `raw_df_square-octagon5.csv` or `df_square-octagon5.csv` : This dataframe[^1] contains all the details of the Square-Octagon lattice like coordinates, nearest neighbor, next-nearest neighbor and next-next-neighbor points. `raw_df_square-octagon5.csv` contains the unformatted dataframe where 4th column is NNN neighbor while `df_square-octagon5.csv` contains the formatted dataframe which structure [R, L, U, D], i.e., For site i, first element is the site position to the right of i, second to the left and so on. This is useful for when constructing the d-wave Hamiltonian.

2. `ham_5_t2_1.0` : This contains the real space tight binding Hamiltonian. One doesn't need this explicitly as this can be made on the fly using the above dataframe. For t2=0.1 the Hamiltonian is `ham_5_t2_0.1` which again can be constructed on the fly. So basically one only needs the dataframe. The source code is agnostic about this and only looks for the Hamiltonian in particular format.

## Workflow 

1. **Step 0 :** Create the necessary directories. data, logs, results alongwith src and scripts
2. **Step 1 :** Run Makefile for fresh calculation. `make fresh` (This will empty the data directory so if you want, take a backup of it else bring it back from `assets` directory)
3. **Step 2 :** Edit the `params.jl` file accordingly
4. **Step 3 :** Run `julia -p <n> script_name.jl` on `n` cores using the script `script_name.jl`

---
[^1]: The dataframe is originally created from a FORTRAN program who sharing rights I do not have. Hence, I have provided the dataframe for lots of lattice sizes in the `assets` directory
