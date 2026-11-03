abstract type Arbre end #type abstrait arbre dont les structure
#de l'arbre vont hériter

struct feuille <: Arbre # une feuille de l'arbre
    prediction::Int
end

struct Noeud <: Arbre # Noeud interne de l'arbre
    indice :: Int # Indice de séparation
    seuil :: Float64 # Seuil de séparation
    gauche :: Arbre#fils(arbre) à gauche
    droite :: Arbre#fils(arbre) à droite
    prediction :: Int #prédiction à un noued donné
end

#=On va se servir des coefficients de gini et/ou l'entropie(ces deux fonctions ont le
même rôle) pour calculer le gain qui est le critère prinvcipale de décision dans les arbres de décision=#

function gini(y::AbstractVector)

    n = length(y)
    if n == 0
        return 0
    end
    p0 = count(==(0),y)/n
    p1 = count(==(1),y)/n
    return 1-p0^2-p1^2
end

function entropie(y::AbstractVector)
    n = length(y)
    p0 = count(y==0,y)/n
    p1 = count(y==1,y)/n 
    if p0 == 0 && p1 !=0
        return -p1*log2(p1)
    elseif p0 !=0 && p1 == 0
        return -p0*log2(p0)
    elseif p0==0 && p1==0
        return 0
    else
        return -p0*log2(p0)-p1*log2(p1)
    end
end

#= La fonction suivante permet de séparer le vecteur de la variable à expliquer
en fonction du vecteur de la variable explicative =#
function separer(A::AbstractVector,y::AbstractVector,seuil::Float64)
    B = y[A.<=seuil]
    C = y[A.>seuil]
    return B,C
end
#=cette nouvelle fonction permet de calculer le coefficient de d'impureté 
après une séparation: elle permettra avec le coefficient de gini de calculer
le gain =#
function impureteGini(fils_gauche::AbstractVector,fils_droite::AbstractVector)
    ng = length(fils_gauche)
    nd = length(fils_droite)
    n = ng+nd
    if n == 0
        return 0
    end

    return (ng*gini(fils_gauche))/n +(nd*gini(fils_droite))/n
end

#=La fonction qui va suivre best_split permet de faire la meilleure
subdivision possible(celle maximisant le gain) pour la variable 
explicative. Elle prend deux arguments (variable explicative,variable à
expliquer) et retourne cette valeur seuil ainsi que la valeur du
gain maximum correspondant à ce seuil =#

function best_split(A::AbstractVector,y::AbstractVector)

    valeurs = sort(unique(A))#trie les valeurs par ordre croissant
    best_gain = -Inf #meilleure gain initial
    best_seuil = nothing # meilleur seuil initial
    
    if length(valeurs)<2
        return -Inf,nothing
    end
    for i in 1:(length(valeurs)-1)

        seuil = (valeurs[i]+valeurs[i+1])/2

        gauche = y[A .<= seuil]
        droite = y[A .> seuil]
        if isempty(gauche)|| isempty(droite)
            continue
        end

        gain = gini(y)-impureteGini(gauche,droite)

        if gain >best_gain
            best_gain = gain
            best_seuil = seuil

        end
    end
    
    return best_gain,best_seuil

end

function meilleure_colonne(X,y)
    best_gain = -Inf
    best_indice = nothing
    best_seuil = nothing
    for i in 1:size(X,2)
        A = X[:,i]
        gain,seuil = best_split(A,y)
        if gain>best_gain
            best_gain = gain
            best_indice = i
            best_seuil = seuil
        end

    end

    return best_gain,best_indice,best_seuil

end