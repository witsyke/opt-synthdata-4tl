using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO
using CSV, DataFrames
using RecursiveArrayTools # for VectorOfArray

# Load the real data 
real_data_df = CSV.read("covid.csv", DataFrame)
# We only have information about the delta of the infections - this we can model as the cumulative sum of the infected
real_data_df.cumulative_sum = cumsum(real_data_df.Infected)
real_data_df
len_real_data = size(real_data_df, 1)
real_data = transpose(real_data_df[1:91, 3]) # convert to necessary format # only use first 90 rows (train + val set in ML)

# -----------------------------------------

# Define the sir ODE system
function sir(du, u, p, t)
    # p[1] = beta, p[2] = gamma 
    # u[1] = suseptible, u[2] = infected, u[3] = recovered, u[4] = cumulative infected
    N = u[1] + u[2] + u[3]
    du[1] = ds = -p[1] * u[1] * u[2] / N
    du[2] = di = p[1] * u[1] * u[2] / N - p[2] * u[2]
    du[3] = dr = p[2] * u[2]
    du[4] = dc = p[1] * u[1] * u[2] / N # cumulative infected
end

# -----------------------------------------

# Define initial conditions
u0 = [84400000.0, 9660, 0, 9660] # initial condition for suseptible, infected, recovered, cumulative infected

# Define time span
tspan = (0.0, 90.0)
t = collect(range(0, stop=90)) # length of calibration window

# Define kinetic parameters
p_init = [0.41, 0.12] # beta, gamma

prob = ODEProblem(sir, u0, tspan, p_init)

sol = solve(prob, Tsit5())
plot(sol)

# -----------------------------------------

# Define cost function for parameter estimation
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t, real_data),
    Optimization.AutoForwardDiff(),
    maxiters=100000, verbose=false, save_idxs=[4])

# Define optimization problem - note the different set of intial parameters - seems rather senstive to the bounds
optprob = Optimization.OptimizationProblem(cost_function, p_init, lb=[0.0, 0.0], ub=[5.0, 0.13])
result_bfgs = solve(optprob, BFGS())

# -----------------------------------------

# Create new problem to solve with the estimated parameters
newprob = remake(prob, tspan=(0, len_real_data), p=result_bfgs.u)
newsol = solve(newprob, Tsit5(), save_idxs=[4])

# -----------------------------------------

t_newsol = collect(range(0, length=len_real_data))
results_df = DataFrame(mapreduce(permutedims, vcat, [(newsol(t_newsol[i])) for i in 1:length(t_newsol)]), [:Infected])
results_df[!, :time] = t_newsol
results_df

# -----------------------------------------

# Plot the estimated solution and the noisy real data
plot(newsol)
plot!(range(0, len_real_data - 1), real_data_df.cumulative_sum)
vline!([90], color=:black)

# -----------------------------------------

CSV.write("sir_ode_forecast_1.csv", results_df)

