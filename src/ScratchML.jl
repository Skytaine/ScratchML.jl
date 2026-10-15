module ScratchML

# Write your package code here.
export distance , init_flags , init_flagsbis

abstract type AbstractModel end

include("logreg.jl")
include("kmean.jl")
#function fit!(m::AbstractModel,X::AbstractVecorMat,y::AbstractVector) un pour chaque modele y existe que dans le cas supervisé
#function predict(m::AbstractModel) 
#function machine(model::AbstractModel,X::DataFrame ou AbstracVecorMat,y::AbstractVector facultatif)

end
