using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO
using CSV, DataFrames
using RecursiveArrayTools # for VectorOfArray

# Load the real data 
real_data_df = CSV.read("lynx-hares.csv", DataFrame)
len_real_data = size(real_data_df, 1)
real_data = transpose(Matrix(real_data_df[1:19, :])) # convert to necessary format # only use first 19 rows (train + val set in ML)

# -----------------------------------------

# Define the lotka volterra ODE system
function lotka_volterra(du, u, p, t)
    # p[1] = alpha, p[2] = beta, p[3] = gamma, p[4] = delta,
    # u[1] = prey, u[2] = predator,
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

# -----------------------------------------

# Define initial conditions
u0 = Array(real_data_df[1, :]) # initial condition for prey and predator

# Define time span
tspan = (0.0, 18)
t = collect(range(0, stop=18))  # length of calibration window

# Define kinetic parameters
p_init = [0.545, 0.028, 0.803, 0.024] # alpha, beta, gamma, delta

prob = ODEProblem(lotka_volterra, u0, tspan, p_init)

# -----------------------------------------

# Define cost function for parameter estimation
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t, real_data),
    Optimization.AutoForwardDiff(),
    maxiters=100000, verbose=false)

# Define optimization problem - note the different set of intial parameters - seems rather senstive to the bounds
optprob = Optimization.OptimizationProblem(cost_function, p_init, lb=[0.0, 0.0, 0.0, 0.0], ub=[10.0, 10.0, 10.0, 10.0])
result_bfgs = solve(optprob, BFGS())

# -----------------------------------------

# Create new problem to solve with the estimated parameters
newprob = remake(prob, tspan=(0, len_real_data), p=result_bfgs.u)
newsol = solve(newprob, Tsit5())

# -----------------------------------------

t_newsol = collect(range(0, length=len_real_data))
results_df = DataFrame(mapreduce(permutedims, vcat, [(newsol(t_newsol[i])) for i in 1:length(t_newsol)]), [:Hare, :Lynx])
results_df[!, :time] = t_newsol
results_df

# -----------------------------------------

# Plot the estimated solution and the noisy real data
plot(newsol)
plot!(range(0, len_real_data - 1), Matrix(real_data_df))
vline!([19], color=:black)

# -----------------------------------------

CSV.write("lynx-hares_ode_forecast.csv", results_df)

# -----------------------------------------

results_windowed = DataFrame(Hare=[], Lynx=[], time=[], index=[])

# Iterate over the test set indices (-5 because we don't have target data for these days)
for i in 19:len_real_data-5

    u0_current = Array(real_data_df[i, :]) # select the ICs from the row corresponing with the text index
    tspan_current = (0.0, 5) # set the length of the solving to 5 timesteps
    t_current = collect(range(0, stop=5)) # collect values for timepoints 0,1,2,3,4,5

    prob_current = remake(prob, u0=u0_current, tspan=tspan_current, p=result_bfgs.u) # set ICs and KPs
    sol_current = solve(prob_current, Tsit5())

    results_df_current = DataFrame(mapreduce(permutedims, vcat, [(sol_current(t_current[i])) for i in 1:length(t_current)]), [:Hare, :Lynx])
    results_df_current[!, :time] = t_current
    results_df_current[!, :index] = [i, i, i, i, i, i]

    results_windowed = [results_windowed; results_df_current]

end

results_windowed
CSV.write("lynx-hares_ode_forecast_windowed.csv", results_windowed)



