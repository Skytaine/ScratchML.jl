module ScratchML
using Random #, Statistics
using DataFrames #pour les dataframes
import StatsAPI: fit!, predict
# Write your package code here.
export distance, init_flags, init_flagsbis , assigned , repo 
export InitAlea, InitChoice
export logvrai, gradient, ajout_col_1, sigm, LogReg, predict


abstract type AbstractModel end

include("logreg.jl")
include("kmean.jl")
#function fit!(m::AbstractModel,X::AbstractVecorMat,y::AbstractVector) un pour chaque modele y existe que dans le cas supervisé
#function predict(m::AbstractModel) 
#function machine(model::AbstractModel,X::DataFrame ou AbstracVecorMat,y::AbstractVector facultatif)

end
