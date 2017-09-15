using DSGE, StateSpaceRoutines
using QuantEcon: solve_discrete_lyapunov
using DataFrames

m = AnSchorfheide()
df = readtable("us.txt", header = false, separator = ' ')
data = convert(Matrix{Float64}, df)'

params = [2.09, 0.98, 2.25, 0.65, 0.34, 3.16, 0.51, 0.81, 0.98, 0.93, 0.19, 0.65, 0.24,
          0.115985, 0.294166, 0.447587]
update!(m, params)

system = compute_system(m)
RRR = system.transition.RRR
QQ = system.measurement.QQ
U, E, V = svd(QQ)
sqrtS2 = RRR*U*diagm(sqrt(E))
TTT = system.transition.TTT

tuning = Dict(:r_star => 2., :c => 0.3, :accept_rate => 0.4, :target => 0.4,
              :xtol => 0., :resampling_method => :systematic, :N_MH => 1,
              :n_particles => 1000, :n_presample_periods => 0,
              :adaptive => true, :allout => true, :parallel => false)

n_states = size(system[:TTT])[1]
s0 = zeros(n_states)
P0 = solve_discrete_lyapunov(TTT, RRR*QQ*RRR')
Φ, Ψ, F_ϵ, F_u = initialize_function_system(system)
s_init = initialize_state_draws(s0, F_ϵ, Φ, tuning[:n_particles])

tempered_particle_filter(data, Φ, Ψ, F_ϵ, F_u, s_init; tuning...)