module Rethinking

using Optim

function map(logmodel, data, parameters_init)
    f(p) = -logmodel(p, data)
    ftd = TwiceDifferentiable(f, parameters_init)

    res = optimize(ftd, parameters_init, BFGS())

    p_map = Optim.minimizer(res)

    numerical_hessian = Optim.hessian!(ftd, p_map)
    cov = inv(numerical_hessian)

    return p_map, cov
end

end # module
