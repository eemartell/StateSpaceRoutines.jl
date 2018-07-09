using BenchmarkTools
using ClusterManagers
using HDF5, JLD, Base.Test
using DataFrames
using DSGE
import DSGE.update!
using QuantEcon: solve_discrete_lyapunov

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

# Generation of the initial state draws
n_states = n_states_augmented(m)
s_0 = zeros(n_states)
P_0 = solve_discrete_lyapunov(T, R*Q*R')

#Kalman
compute_values(data, T, R, C, Q, Z, D, E, s_0, P_0)
