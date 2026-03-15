N = 5
nSites = 4 * N^2
lattice = "square-octagon"

t2 = 1.0
lambda = 0.0
isoMap = nothing

mu = -1.0
U  = 3.0

J = 0.0
K = 0.0
T = 0.0

impuritySite = 3

twist = 1e-3
maxCount = 200
tol      = 1e-3

# lambdaVals = range(0.0, 0.4; step=0.025)
# muVals = range(-3, stop=4, length=11)
lambdaVals = [0.1, 0.2]
muVals = [0.0, -0.5]
UVals = [1.0, 3.0]
t2Vals = [0.1, 1.0]
TVals = [0.0, 0.1]
NVals = [5, 10]

# Initial guesses (kept here as requested)
# NOTE: main(λ) will pick Float64 vs ComplexF64 to match λ; these are defaults.
nUp0     = ones(Float64, nSites)
nDn0     = ones(Float64, nSites)
deltaOld0 = ones(ComplexF64, nSites)   # default; for λ=0 we’ll use Float64 internally

dataSetFolder = "../data/"
logFolder     = "../logs/"
saveInFolder  = "../results/"

rawdfName    = "$(dataSetFolder)raw_df_$(lattice)$(N).csv"
tMatFileName = "$(dataSetFolder)ham_$(N)_t2_$(t2)"

includeHartree = false
lambdaIso = 0.0 # not properly implemented; not checked
