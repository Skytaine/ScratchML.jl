module ScratchML
using Random #, Statistics
using DataFrames #pour les dataframes
# Write your package code here.
export distance, init_flags, init_flagsbis
export InitAlea, InitChoice
export fit!, predict, logvrai, gradient, ajout_col_1, sigm, LogReg


abstract type AbstractModel end

include("logreg.jl")
include("kmean.jl")
#function fit!(m::AbstractModel,X::AbstractVecorMat,y::AbstractVector) un pour chaque modele y existe que dans le cas supervisé
#function predict(m::AbstractModel) 
#function machine(model::AbstractModel,X::DataFrame ou AbstracVecorMat,y::AbstractVector facultatif)

end
