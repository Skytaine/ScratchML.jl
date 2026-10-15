#Notre package
using ScratchML
#Package pour comparer
using MLJ
#Pour utiliser des dataframe et csv
using DataFrames
using CSV
#Pour avoir des dataframes exemples usuelles
using RDatasets

using Symbolics

n, p = 100, 3
Xb = hcat(ones(n), randn(n, p))
y  = rand([0.0, 1.0], n)

function grad_num(w, X, y; h=1e-6)
    g = zeros(length(w))
    for j in eachindex(w)
        e = zeros(length(w)); e[j] = h
        g[j] = (ScratchML.logvrai(w .+ e, X, y) - ScratchML.logvrai(w .- e, X, y)) / (2h)
    end
    g
end

w = randn(p + 1)
ScratchML.gradient(w, Xb, y) ≈ grad_num(w, Xb, y)   # doit renvoyer true