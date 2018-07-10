using JLD, HDF5
using QuantEcon: solve_discrete_lyapunov

state_space_vars = load("state_space_vars.jld")
T = state_space_vars["T"]
R = state_space_vars["R"]
C = vec(state_space_vars["C"])

include("test.jl")
include("kalman_filter.jl")

convert = 13
dt = 1/convert

#grab inflation series
data_orig = h5read("smc.h5","data")[2,:]

nt = length(data_orig)
Nt = convert*nt

Q = eye(1,1)*dt
E = zeros(1,1)
D = [3.69]
Z = zeros(401,1)
Z[201] = 400

data = fill(NaN, Nt, 1)
for i in 1:Nt
    if i%13 == 1
        data[i] = data_orig[Int(ceil(i/13))]
    end
end

T_tilde = T*dt + eye(size(T,1))

n_states = 401
s_0 = zeros(n_states)
P_0 = 1e6*eye(n_states) #solve_discrete_lyapunov(T_tilde, R*Q*R')

#Kalman
kalman_filter(data, T_tilde, R, C, Q, Z, D, E, s_0, P_0)

