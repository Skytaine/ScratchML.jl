
#on définit le modèle de régréssion linéaire

mutable struct LogReg <: AbstractModel
    weights::Vector #Poids de la régression logistique
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

#je veux utiliser des matrices pour les calculs mais X est un DataFrames
#J'utilise le multiple dispatching sur la fonction qui ajoute la colonne de 1 pour utiliser fit et predict avec dataframe et matrices
function ajout_col_1(X::AbstractMatrix)
    Xb = hcat(ones(size(X, 1)), X)
    return Xb
end

function ajout_col_1(X::AbstractDataFrame)
    X=Matrix(X)
    return ajout_col_1(X)
end

#fonction fit! : la descente de gradient
function fit!(model::LogReg,X,y::Vector)
    n=size(X,1) #on divise le gradient pour pas avoir de probleme avec le pas en fonction de n
    Xb = ajout_col_1(X)   # ajoute une colonne de 1 (colonne pour le coefficient constant dans weights)
    w=zeros(size(Xb,2)) #vecteur weights avec que des zéros
    for i in 1:model.max_iter #on limite le nombre d'itération si on diverge on converge trop lentement
        g=gradient(w,Xb,y) ./n
        if sqrt(sum(g .^2))<model.tol #si on converge vers le max on s'arrete
            break
        end
        w=w + model.lr .* g
    end
    model.weights=w
    print("Fit LogReg fini")
    return model
end

#fonction predict
function predict(model::LogReg,X)
    p=sigm.(ajout_col_1(X)*model.weights)
    return p .> 0.5
end


