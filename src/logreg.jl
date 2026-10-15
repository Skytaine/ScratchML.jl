#on définit le modèle de régréssion linéaire

mutable struct LogReg <: AbstractModel
    weights::Vector{Float64} #Poids de la régression logistique
    max_iter::Int64 #on limite le nombre d'itération pour converger vers le meilleurs paramètres
    tol::Float64 #le seuil d'ou on considère que la descente de gradient a converger
    lr::Float64 #les pas de la descente de gradient
end

#son constructeur
LogReg(;weights=Float64[],max_iter=100,tol=1e-16,lr=0.1)=LogReg(weights,max_iter,tol,lr)



#Fonction logistique
function sigm(a::Real)
    1/(1+exp(-a))
end

#Fonction LogVraisemblance
#potentielle probleme si p=0 ou 1 devrait pas arrivé
function logvrai(w::Vector,X::Matrix,y::Vector)
    #on a théoriquement n variable explicative et n+1 poids
    #ao est un poids constant dans la sigmoide
    #on par du principe qu'on rajoute une colonne de 1 à la matrice de donnée
    p=sigm.(X*w)
    sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
end


#On va maximiser pour une valeur du vecteur weights la logvrai
#Fonction gradient
function gradient(w::Vector,X::Matrix,y::Vector)
    p=sigm.(X*w)
    X'*(y .- p)
end

#fonction fit! : la descente de gradient
function fit!(model::LogReg,X,y::Vector)
    
end