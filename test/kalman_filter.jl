using DSGE, HDF5, Base.Test
include("../util.jl")

path = dirname(@__FILE__)

# Initialize arguments to function
h5 = h5open("$path/../reference/kalman_filter_args.h5")
for arg in ["data", "TTT", "RRR", "CCC", "QQ", "ZZ", "DD", "MM", "EE", "z0", "P0"]
    eval(parse("$arg = read(h5, \"$arg\")"))
end
close(h5)

# Method with all arguments provided
out = kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, MM, EE, z0, P0)

h5 = h5open("$path/reference/kalman_filter_out.h5")
@test_approx_eq read(h5, "L")             out[1]
@test_approx_eq read(h5, "pred")          out[2]
@test_approx_eq read(h5, "vpred")         out[3]
@test_approx_eq read(h5, "yprederror")    out[4]
@test_approx_eq read(h5, "ystdprederror") out[5]
@test_approx_eq read(h5, "rmse")          out[6]
@test_approx_eq read(h5, "rmsd")          out[7]
@test_approx_eq read(h5, "filt")          out[8]
@test_approx_eq read(h5, "vfilt")         out[9]
close(h5)

# Method with initial conditions omitted
out = kalman_filter(data, TTT, RRR, CCC, QQ, ZZ, DD, MM, EE)

# Pend, vpred, and vfilt matrix entries are especially large, averaging 1e5, so
# we allow greater ϵ
h5 = h5open("$path/reference/kalman_filter_out.h5")
@test_approx_eq     read(h5, "L")             out[1]
@test_approx_eq     read(h5, "pred")          out[2]
@test_approx_eq_eps read(h5, "vpred")         out[3]  1e-1
@test_approx_eq     read(h5, "yprederror")    out[4]
@test_approx_eq     read(h5, "ystdprederror") out[5]
@test_approx_eq     read(h5, "rmse")          out[6]
@test_approx_eq     read(h5, "rmsd")          out[7]
@test_approx_eq     read(h5, "filt")          out[8]
@test_approx_eq_eps read(h5, "vfilt")         out[9] 1e-1
close(h5)

nothing