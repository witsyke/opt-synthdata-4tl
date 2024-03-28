using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO
using CSV, DataFrames
using RecursiveArrayTools # for VectorOfArray

coherence = "coherent" #incoherent

# Load the real data 
real_data_df = CSV.read("algae-rotifers-" * coherence * ".csv", DataFrame)
# Scale real data to correct units
real_data_df[!, 2] = real_data_df[!, 2] * 10^9
real_data_df[!, 3] = real_data_df[!, 3] * 10^3
len_real_data = size(real_data_df, 1)
real_data = transpose(Matrix(real_data_df[1:7, 2:3])) # convert to necessary format # only use first 7 rows (train + val set in ML)


# -----------------------------------------

# Define the algae rotifiers ODE system
function algae_rotifiers(du, u, p, t)
    # p[1] = delta, p[2] = S* p[3] = ca, p[4] = fa, p[5] = ha, p[6] = cr, p[7] = fr, p[8] = hr
    # u[1] = nitrogen, u[2] = algae, u[3] = rotifers
    du[1] = ds = (p[2] - u[1]) * p[1] - (1 / p[3]) * (p[4] * u[1] / (p[5] + u[1])) * u[2]
    du[2] = da = ((p[4] * u[1]) / (p[5] + u[1])) * u[2] - (1 / p[6]) * ((p[7] * u[2]) / (p[8] + u[2])) * u[3] - p[1] * u[2]
    du[3] = dr = (p[7] * u[2]) / (p[8] + u[2]) * u[3] - p[1] * u[3]
end

# -----------------------------------------

# Define initial conditions
u0 = vcat([10], Array(real_data_df[1, 2:3])) # initial condition for nitorgen, algae, rotifers

# Define time span
tspan = (0.0, 6)
t = collect(range(0, stop=6)) # length of calibration window

# Define kinetic parameters
p_init = [0.55, 80, exp(17.68), exp(1.38), 4.3, exp(-11.82), exp(0.23), 7.5 * 10^8] # delta, S*, ca, fa, ha, cr, fr, hr

prob = ODEProblem(algae_rotifiers, u0, tspan, p_init)

sol = solve(prob, AutoTsit5(Rosenbrock23()), save_idxs=[2])
plot(sol)

# -----------------------------------------

# This is to ensure that algeae and rotifiers are on the same scale for the loss function

data_weights = repeat([1.0 * 10^3, 1.0 * 10^9], 1, 7)
# Define cost function for parameter estimation
cost_function = build_loss_objective(prob, AutoTsit5(Rosenbrock23()), L2Loss(t, real_data, data_weight=data_weights),
    Optimization.AutoForwardDiff(),
    maxiters=100000, verbose=false, save_idxs=[2, 3])

# Define optimization problem - note the different set of intial parameters - seems rather senstive to the bounds
optprob = Optimization.OptimizationProblem(cost_function, p_init, lb=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ub=[10^30, 10^30, 10^30, 10^30, 10^30, 10^30, 10^30, 10^30])
result_bfgs = solve(optprob, BFGS(linesearch=LineSearches.BackTracking()))

# -----------------------------------------

# Create new problem to solve with the estimated parameters
newprob = remake(prob, tspan=(0, len_real_data), p=result_bfgs.u)
newsol = solve(newprob, AutoTsit5(Rosenbrock23()), save_idxs=[2, 3])

# -----------------------------------------

t_newsol = collect(range(0, length=len_real_data))
results_df = DataFrame(mapreduce(permutedims, vcat, [(newsol(t_newsol[i])) for i in 1:length(t_newsol)]), [:Algae, :Rotifers])
results_df[!, :time] = t_newsol
results_df

# -----------------------------------------

# Plot the estimated solution and the noisy real data
plot(newsol)
plot!(range(0, len_real_data - 1), real_data_df.algae)
plot!(range(0, len_real_data - 1), real_data_df.rotifers)
vline!([6], color=:black)

# -----------------------------------------

CSV.write("algae_rotifiers_" * coherence * "_ode_forecast.csv", results_df)
