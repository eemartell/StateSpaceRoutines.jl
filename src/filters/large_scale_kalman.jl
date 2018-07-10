# This file implements the block Kalman filter from Strid and Walentin 2008
# and an ensemble filter.
#
# Notes on block filter for user's convenience:
# If S is the state vector, then
# S' = [S_1', S_2' S_3' S_4'],
# where S_1 contains exogenous AR(1) processes,
# S_2 contains exogenous VAR(1) processes,
# S_3 contains exogenous variables that appear in the observation equation,
# and S_4 contains endogenous variables.
#
# Suppose \tilde{S} is the original un-ordered matrix.
# It is possible to define re-ordering matrix M, with M_{ij} = 1 if the
# variable in place i in \tilde{S} obtains the position j in S, 0 otherwise.
# Then the ordered model is obtained via
# T = M' \tilde{T} M, R = M' \tilde{R} \tilde{M},
# Q = \tilde{M}' \tilde{Q} \tilde{M}, Z = \tilde{Z} M,
# where \tilde{M} is a submatrix of M obtained by deleting rows/columns
# associated with endogenous variables.
#
# We additionally add the option of specifying which block filter
# we want, with the default being a 2-block (1 and 4) filter, and
# the Boolean true_block indicates whether the system is placed
# in the exact format to optimally use the block filter, e.g.
# all AR(1) shocks are represented as states
include("kalman_filter.jl")

function block_kalman_filter(y::Matrix{Float64}, Ttild::Matrix{Float64}, Rtild::Matrix{Float64},
                             Ctild::Vector{Float64}, Qtild::Matrix{Float64}, Ztild::Matrix{Float64},
                             Dtild::Vector{Float64}, Htild::Matrix{Float64},
                             M::Matrix{Float64}, Mtild::Matrix{Float64}, block_dims::Vector{Int64},
                             s_0tild::Vector{Float64} = Vector{Float64}(0),
                             P_0tild::Matrix{Float64} = Matrix{Float64}(0,0);
                             outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                             Nt0::Int = 0, block_num::Int64 = 2, true_block::Bool = false)
    # Re-order matrices and vectors
    T = M' * Ttild * M
    R = M' * Rtild * Mtild
    C = M * Ctild
    Q = Mtild' * Qtild * Mtild
    Z = Ztild * M
    D = Dtild
    H = Htild
    s_0 = M * s_0tild
    P_0 = M' * P_0tild * M

    # check for whether the matrices satisfy block-kalman filter set up
    if block_num == 2 && Z[:, 1:block_dims[1]] != zeros(size(Z, 1), block_dims[1])
        # In true_block case, Z1 = 0
        true_block = false
    elseif block_num == 2 && block_dims[1] != size(R, 2)
        # In true 2-block case, number of exogenous shocks equals sum of first three block entries
        true_block = false
    end

    # Determine outputs
    return_loglh = :loglh in outputs
    return_pred  = :pred in outputs
    return_filt  = :filt in outputs

    # Dimensions
    Ns = size(T,1) # number of states
    Nt = size(y,2) # number of periods of data

    # Initialize Inputs and outputs
    k = KalmanFilter(T, R, C, Q, Z, D, H, s_0, P_0)

    mynan  = convert(Float64, NaN)
    loglh  = return_loglh ? fill(mynan, Nt)         : Vector{Float64}(0)
    s_pred = return_pred  ? fill(mynan, Ns, Nt)     : Matrix{Float64}(0, 0)
    P_pred = return_pred  ? fill(mynan, Ns, Ns, Nt) : Array{Float64, 3}(0, 0, 0)
    s_filt = return_filt  ? fill(mynan, Ns, Nt)     : Matrix{Float64}(0, 0)
    P_filt = return_filt  ? fill(mynan, Ns, Ns, Nt) : Array{Float64, 3}(0, 0, 0)

    # Populate initial states
    s_0 = k.s_t
    P_0 = k.P_t

    # Loop through periods t
    for t = 1:Nt
        # Forecast
        forecast!(k, block_num, block_dims, true_block)
        if return_pred
            s_pred[:,    t] = k.s_t
            P_pred[:, :, t] = k.P_t
            try
                @assert isapprox(k.s_t, s_pred_true[:, t])
                @assert isapprox(k.P_t, P_pred_true[:, :, t])
            catch
                println("Failed at forecast, time step $t")
                @assert false
            end
        end

        # Update and compute log-likelihood
        update!(k, y[:, t], block_num, block_dims, true_block; return_loglh = return_loglh)
        if return_filt
            s_filt[:,    t] = k.s_t
            P_filt[:, :, t] = k.P_t
            try
                @assert isapprox(k.s_t, s_filt_true[:, t])
                @assert isapprox(k.P_t, P_filt_true[:, :, t])
            catch
                println("Failed at update, time step $t")
                @assert false
            end
        end
        if return_loglh
            loglh[t]        = k.loglh_t
        end

        # Update s_0 and P_0 if Nt_0 > 0, presample periods
        if t == Nt0
            s_0 = k.s_t
            P_0 = k.P_t
        end
    end

    # Populate final states
    s_T = k.s_t
    P_T = k.P_t

    # Remove presample periods
    loglh, s_pred, P_pred, s_filt, P_filt =
        remove_presample!(Nt0, loglh, s_pred, P_pred, s_filt, P_filt; outputs = outputs)

    return loglh, s_pred, P_pred, s_filt, P_filt, s_0, P_0, s_T, P_T
end

# Computes the one-step-ahead states s_{t|t-1} and state covariances P_{t|t-1}
# and assign to 'k'
function forecast!(k::KalmanFilter{Float64}, block_num::Int64, block_dims::Vector{Int64}, true_block::Bool)
    T, R, C, Q = k.T, k.R, k.C, k.Q
    s_filt, P_filt = k.s_t, k.P_t

    if block_num == 2
        # Get block matrices that don't depend on true_block
        dim1 = block_dims[1] # dimension of exogenous AR(1) processes
        dim2 = block_dims[4] # dimension of endogenous states
        s1_filt  = s_filt[1:dim1]
        s2_filt  = s_filt[dim1 + 1:end]
        P11_filt = P_filt[1:dim1, 1:dim1]
        P12_filt = P_filt[1:dim1, dim1 + 1:end]
        P22_filt = P_filt[dim1 + 1:end, dim1 + 1:end]
        Ablock = T[1:dim1, 1:dim1]
        Cblock = T[dim1 + 1:end, dim1 + 1:end]

        # Compute P_t, s_t
        if true_block
            Bblock = R[dim1 + 1:end, :]
            L_t = (P12_filt * Cblock') .* (diag(Ablock) * ones(1, dim2))
            P11_t = P11_filt .* (diag(Ablock) * diag(Ablock)') + Q
            P12_t = P11_t * Bblock' + L_t
            G = Bblock * (P12_t + L_t)
            P22_t = (G + G')/2 + Cblock * P22_filt * Cblock'

            s1_t = diag(Ablock) .* s1_filt
            s2_t = Bblock * s1_t + Cblock * s2_filt
        else
            Bblock = R[dim1 + 1:end, 1:dim1]
            Rup = R[1:dim1, dim1 + 1:end] # remaining columns of R matrix
            Rlo = R[dim1 + 1:end, dim1 + 1:end]
            Q1 = Q[1:dim1, 1:dim1]
            Q2 = Q[dim1 + 1:end, dim1 + 1:end]

            # RQR' = |  Q1 + Rup * Q2 * Rup', Q1*Bblock' + Rup * Q2 * Rlo'        |
            #        |     ---           , Bblock * Q1 * Bblock' + Rlo * Q2 * Rlo'|
            L_t = (P12_filt * Cblock') .* (diag(Ablock) * ones(1, dim2))
            RupQ2 = Rup * Q2

            # Compute intermediate matrices to minimize additions
            P11_t = P11_filt .* (diag(Ablock) * diag(Ablock)') + Q1
            P12_t = P11_t * Bblock' + L_t
            G = Bblock * (P12_t + L_t)

            # Compute final matrices
            P22_t = (G + G')/2 + Rlo * Q2 * Rlo' + Cblock * P22_filt * Cblock'
            P12_t = P12_t + RupQ2 * Rlo'
            P11_t = P11_t + RupQ2 * Rup'

            s1_t = diag(Ablock) .* s1_filt
            s2_t = Bblock * s1_t + Cblock * s2_filt
        end

        # Save values
        k.s_t = vcat(s1_t, s2_t)
        k.P_t = [P11_t P12_t; P12_t' P22_t]
    end

    return nothing
end

function update!(k::KalmanFilter{Float64}, y_obs::Vector{Float64}, block_num::Int64, block_dims::Vector{Int64},
                 true_block::Bool; return_loglh::Bool = true)
    # Keep rows of measurement equation corresponding to non-NaN observables
    nonnan = .!isnan.(y_obs)
    y_obs = y_obs[nonnan]
    Z = k.Z[nonnan, :]
    D = k.D[nonnan]
    H = k.E[nonnan, nonnan]
    Ny = length(y_obs)

    s_pred = k.s_t
    P_pred = k.P_t

    if block_num == 2
        # Get block matrices
        dim1 = block_dims[1] # dimension of exogenous AR(1) processes
        dim2 = block_dims[4] # dimension of endogenous states
        s1_pred  = s_pred[1:dim1]
        s2_pred  = s_pred[dim1 + 1:end]
        P11_pred = P_pred[1:dim1, 1:dim1]
        P12_pred = P_pred[1:dim1, dim1 + 1:end]
        P22_pred = P_pred[dim1 + 1:end, dim1 + 1:end]
        Z2 = Z[:, dim1 + 1:end]

        # Compute predicted y, measurement covariance matrix, error, and auxiliary matrices
        if true_block
            y_pred = Z2 * s2_pred + D
            V_pred = Z2 * P22_pred * Z2' + H
        else
            Z1 = Z[:, 1:dim1]
            off_diag_mat = Z1 * P12_pred * Z2'
            y_pred = Z * s_pred + D
            V_pred = Z1 * P11_pred * Z1' + off_diag_mat + off_diag_mat' + Z2 * P22_pred * Z2' + H
        end
        V_pred_inv = inv(V_pred)
        dy = y_obs - y_pred # prediction error

        # Update state covariance matrix
        if true_block
            PZV_1 = P12_pred * Z2' * V_pred_inv
            PZV_2 = P22_pred * Z2' * V_pred_inv

            Z2_P22_pred = Z2 * P22_pred
            P11_t = P11_pred - PZV_1 * Z2 * P12_pred'
            P12_t = P12_pred - PZV_1 * Z2_P22_pred
            P22_t = P22_pred - PZV_2 * Z2_P22_pred
        else
            kalman_gain_1 = P11_pred * Z1' + P12_pred * Z2'
            kalman_gain_2 = P12_pred' * Z1' + P22_pred * Z2'
            PZV_1 = kalman_gain_1 * V_pred_inv
            PZV_2 = kalman_gain_2 * V_pred_inv

            P11_t = P11_pred - PZV_1 * kalman_gain_1'
            P12_t = P12_pred - PZV_1 * kalman_gain_2'
            P22_t = P22_pred - PZV_2 * kalman_gain_2'
        end

        # Update states
        s1_t = s1_pred + PZV_1 * dy
        s2_t = s2_pred + PZV_2 * dy

        # Save matrices
        k.s_t = vcat(s1_t, s2_t)
        k.P_t = [P11_t P12_t; P12_t' P22_t]

        if return_loglh
            k.loglh_t = -(Ny * log(2*π) + log(det(V_pred)) + dy'*V_pred_inv*dy)/2
        end
    end

end