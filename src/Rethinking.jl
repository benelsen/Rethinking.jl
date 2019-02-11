module Rethinking

using Optim, Format, LinearAlgebra, StatsBase, Distributions, Parameters, DataFrames

abstract type Model end
abstract type Model2 end

function map(m::Model, μ0)
    data = m.d

    f(p) = -m(p, data)
    ftd = TwiceDifferentiable(f, μ0)

    res = optimize(ftd, μ0, BFGS())

    μ = Optim.minimizer(res)

    numerical_hessian = Optim.hessian!(ftd, μ)
    Σ = Matrix(Hermitian( inv(numerical_hessian) ))

    dist = MvNormal(μ, Σ)

    return dist
end

function map(m::Model2, θ0 = nothing; init = :mean)
    @unpack data, priors = m

    if θ0 == nothing
        if init == :mean
            θ0 = [mean(v) for (k, v) in priors]
        elseif init == :random
            θ0 = [rand(v) for (k, v) in priors]
        end
    end

    f(θ) = -m(θ)
    ftd = TwiceDifferentiable(f, θ0)

    res = optimize(ftd, θ0, BFGS(), Optim.Options(show_trace = false, iterations = 1000, extended_trace = false, show_every = 1))

    # @show res

    μ = Optim.minimizer(res)

    numerical_hessian = Optim.hessian!(ftd, μ)

    return μ, numerical_hessian

    Σ = Matrix(Hermitian( pinv(numerical_hessian) ))

    dist = MvNormal(μ, Σ)

    return dist
end

function link(m::Model2, d::Distribution, data = nothing; n = 10_000)
    samples_ = rand(d, n)'
    samples = [samples_[:,i] for i in 1:size(samples_, 2)]

    @unpack link = m

    if data == nothing
        @unpack data = m
    end

    μs = [link(samples, case) for case in eachrow(data)]
end

function simulate(m::Model2, d::Distribution, data = nothing; n = 10_000)
    samples_ = rand(d, n)
    samples = [samples_[i,:] for i in 1:size(samples_, 1)]

    @unpack link = m

    if data == nothing
        @unpack data = m
    end

    ys = [
        rand.( Normal.( link(samples, case), samples_[length(d),:] ) )
        for case in eachrow(data)
    ]
end

function precis(d::Distribution; names = nothing)
    μ = mean(d)
    Σ = cov(d)
    ρ = cor(d)

    if names == nothing
        names = string.(1:size(d)[1])
    end

    varnamelength = maximum(length.(names))

    printfmt("{:<$(varnamelength)s}", "p")
    print("        Mean      StdDev        5.5%       94.5%")
    for i in 1:length(μ)
        printfmt(" {:>11s}", names[i])
    end
    println("")
    for i in 1:length(μ)

        ud = Normal(μ[i], sqrt(Σ[i,i]))

        printfmt("{1:<$(varnamelength)s} {2:s} {3:s} {4:s} {5:s} ",
            names[i],
            num_fmt(μ[i]),
            num_fmt(sqrt(Σ[i,i])),
            num_fmt(quantile(ud, 0.055)),
            num_fmt(quantile(ud, 1-0.055)))

        for j in 1:length(μ)
            printfmt("{:s} ", num_fmt(ρ[i,j]))
        end
        println("")
    end
end

function num_fmt(x; fmte = "{:11.3e}", fmtf = "{:11.3f}", low = 5e-2, high = 1e7)
    abs(x) < low || abs(x) >= high ? format(fmte, x) : format(fmtf, x)
end

end # module
