using BenchmarkTools
using ClusterManagers
using HDF5, JLD, Base.Test
using DataFrames
using DSGE
import DSGE.update!
using QuantEcon: solve_discrete_lyapunov, solve_discrete_riccati

include("test.jl")
include("kalman_filter.jl")

m = AnSchorfheide()

data = h5read("smc.h5","data")
params = [2.09, 0.98, 2.25, 0.65, 0.34, 3.16, 0.51, 0.81, 0.98, 0.93, 0.19, 0.65, 0.24,
          0.115985, 0.294166, 0.447587]
update!(m, params)

# Solution to a Linear DSGE Model w/ IID Gaussian Errors

system  = compute_system(m)
T     = system[:TTT]
R     = system[:RRR]
C     = system[:CCC]
Q      = system[:QQ]
Z      = system[:ZZ]
D      = system[:DD]
E      = system[:EE]
println(size(Q))
@assert false

#test stability of T (if T [in their notation F] according to http://webee.technion.ac.il/people/shimkin/Estimation09/ch6_ss.pdf THM 1 P approaches Pbar and Pbar is the unique non-negative-definite solution of the Algebraic Riccati Equation) For stability |λ_i| < 1 for all i
if all(eigvals(T) .< 1)
    println("T is stable.")
    #Pbar = solve_discrete_riccati(A, B, R, Q, N) see http://quantecon.github.io/QuantEcon.jl/latest/api/QuantEcon.html and compare inputs to above link
else
    println("T is not stable.")
end

# Generation of the initial state draws
n_states = n_states_augmented(m)
s_0 = zeros(n_states)
P_0 = solve_discrete_lyapunov(T, R*Q*R')

#Kalman
norm_P_T, ch_ll, truelik = compute_values(data, T, R, C, Q, Z, D, E, s_0, P_0)

kaloutput = Dict{Symbol, Any}()

kaloutput[:norm_P_T] = norm_P_T
kaloutput[:ch_ll] = ch_ll
kaloutput[:truelik] = truelik