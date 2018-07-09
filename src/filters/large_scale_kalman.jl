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
# we want, with the default being a 2-block (1 and 4) filter.
include("kalman_filter.jl")

function block_kalman_filter(y::Matrix{S}, Ttild::Matrix{S}, Rtild::Matrix{S},
                             Ctild::Vector{S}, Qtild::Matrix{S}, Ztild::Matrix{S},
                             Dtild::Vector{S}, Htild::Matrix{S},
                             s_0tild::Vector{S} = Vector{S}(0),
                             P_0tild::Matrix{S} = Matrix{S}(0,0),
                             M::Matrix{S}, Mtild::Matrix{S}, block_dims::Vector{Int64};
                             outputs::Vector{Symbol} = [:loglh, :pred, :filt],
                             Nt0::Int = 0, block_num::Int64 = 2) where {S<:Abstract Float}
    # Re-order matrices and vectors
    T = M' * Ttild * M
    R = M' * Rtild * Mtild
    C = M * Ctild
    Q = Mtild' * Qtild * Mtild
    Z = Ztild * M
    D = Dtild
    H = Htild
    s_0 = M * s_0tild
    P_0 = M' * P_0tild * Mtild

    # Determine outputs
    return_loglh = :loglh in outputs
    return_pred  = :pred in outputs
    return_filt  = :filt in outputs

    # Dimensions
    Ns = size(T,1) # number of states
    Nt = size(y,2) # number of periods of data

    # Initialize Inputs and outputs
    k = KalmanFilter(T, R, C, Q, Z, D, H, s_0, P_0)

    mynan  = convert(S, NaN)
    loglh  = return_loglh ? fill(mynan, Nt)         : Vector{S}(0)
    s_pred = return_pred  ? fill(mynan, Ns, Nt)     : Matrix{S}(0, 0)
    P_pred = return_pred  ? fill(mynan, Ns, Ns, Nt) : Array{S, 3}(0, 0, 0)
    s_filt = return_fil   ? fill(mynan, Ns, Nt)     : Matrix{S}(0, 0)
    P_filt = return_filt  ? fill(mynan, Ns, Ns, Nt) : Array{S, 3}(0, 0, 0)

    # Populate initial states
    s_0 = k.s_t
    P_0 = k.P_t

    # Loop through periods t
    for t = 1:Nt
        # Forecast
        forecast!(k, block_num)
        if return_pred
            s_pred[:,    t] = k.s_t
            P_pred[:, :, t] = k.P_t
        end

        # Update and compute log-likelihood
        update!(k, y[:, t], block_num; return_loglh = return_loglh)
        if return_filt
            s_filt[:,    t] = k.s_t
            P_filt[:, :, t] = k.P_t
        end
        if return_loglh
            loglh[t]        = k.loglh_t
        end

        # Update s_0 and P_0 if Nt_0 > 0, presample periods
        if t == Nt_0
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

# Computes the on-step-ahead states s_{t|t-1} and state covariances P_{t|t-1}
# and assign to 'k'
function forecast!(k::KalmanFilter{S}, block_num::Int64) where {S<:AbstractFloat}
    T, R, C, Q = k.T, k.R, k.C, k.Q
    s_filt, P_filt = k.s_t, k.P_t

    if block_num == 2
        # Get block matrices
        dim1 = block_dims[1] # dimension of exogenous AR(1) processes
        dim2 = block_dims[4] # dimension of endogenous states
        s1_filt  = s_filt[1:dim1]
        s2_filt  = s_filt[dim1 + 1:end]
        P11_filt = P_filt[1:dim1, 1:dim1]
        P12_filt = P_filt[1:dim1, dim1 + 1:end]
        P22_filt = P_filt[dim1 + 1:end, dim1 + 1:end]
        Ablock = T[1:dim1, 1:dim1]
        Bblock = R[dim1 + 1:end, dim1 + 1:end]
        Cblock = T[dim1 + 1:end, dim1 + 1:end]
        Q1 = Q[1:dim1, 1:dim1]
        Q2 = Q[dim1 + 1:end, dim1 + 1:end]

        # Compute P_t
        L_t = (P12_filt * Cblock') .* diag(Ablock) * ones(1, dim2)
        P11_t = P11_filt .* diag(Ablock) * diag(Ablock') + Q
        P12_t = P11_t * Bblock' + L_t
        G = Bblock *(P12_t + L_t)
        P22_t = (G + G')/2 + Cblock * P22_filt * Cblock'

        # Compute s_t
        s1_t = diag(Ablock) .* s1_filt
        s2_t = Bblock * s1_t + Cblock * s2_filt

        # Save values
        k.s_t = vcat(s1_t, s2_t)
        k.P_t = [P11_t P12_t; zeros(dim2, dim1) P22_t]
    end

    return nothing
end

function update!(k::KalmanFilter{S}, y_obs::Vector{S}, block_num::Int64;
                 return_loglh::Bool = true) where {S<:AbstractFloat}
    # Keep rows of measurement equation corresponding to non-NaN observables
    nonnan = .!isnan.(y_obs)
    y_obs = y_obs[nonnan]
    Z = k.Z[nonnan, :]
    D = k.D[nonnan]
    E = k.E[nonnan, nonnan]
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
        Z2 = Z[dim1 + 1:end]

        # Compute predicted y, measurement covariance matrix, error, and auxiliary matrices
        y_pred = Z2 * s2_pred + D
        V_pred = Z2 * P22_pred * Z2' + H # Z = [0 Z2]
        V_pred_inv = inv(V_pred)
        dy = y_obs - y_pred # prediction error
        PZV_1 = P12_pred * Z2' * V_pred_inv
        PZV_2 = P22_pred * Z2' * V_pred_inv

        # Update states
        s1_t = s1_pred + PZV_1 * dy
        s2_t = s2_pred + PZV_2 * dy

        # Update state covariance matrix
        Z2_P22_pred = Z2 * P22_pred
        P11_t = P11_pred - PZV_1 * Z2 * P12_pred'
        P12_t = P12_pred - PZV_1 * Z2_P22_pred
        P22_t = P22_pred - PZV_2 * Z2_P22_pred

        # Save matrices
        k.s_t = vcat(s1_t, s2_t)
        k.P_t = [P11_t P12_t; zeros(dim2, dim1) P22_t]

        if return_loglh
            k.loglh_t = -(Ny * log(2*π) + log(det(V_pred)) + dy'*V_pred_inv*dy)/2
        end
    end

end