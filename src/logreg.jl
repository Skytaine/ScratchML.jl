#on définit le modèle de régréssion linéaire

mutable struct LogReg <: AbstractModel
    weights::Vector{Float64} #Poids de la régression logistique
    max_iter::Int64 #on limite le nombre d'itération pour converger vers le meilleurs paramètres
end

#son constructeur
LogReg(;weights=[0],max_iter=100)=LogReg(weights,max_iter)

