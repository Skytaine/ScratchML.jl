mutable struct LogReg <: AbstractModel
    weights::Vector{Float64} #Poids de la régression logistique
end

LogReg(;weights=[0])=LogReg()