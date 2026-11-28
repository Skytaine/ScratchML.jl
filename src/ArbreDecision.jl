abstract type Arbre end #type abstrait arbre dont les structure
#de l'arbre vont hériter

struct Feuille <: Arbre # une feuille de l'arbre
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

function gini(y::AbstractVector) # calcule l'indice de gini pour deux classe

    n = length(y)
    if n == 0
        return 0
    end
    p0 = count(==(0),y)/n
    p1 = count(==(1),y)/n
    return 1-p0^2-p1^2
end

function giniG(y::AbstractVector) # calcule l'indice de gini pour n classes
    n = length(y)
    if n == 0
        return 0
    end
    modalite = unique(y) #fonction julia donnant les modalités
    res = 0.0
    for mi in modalite
        res = res + (count(==(mi),y)/n)^2
    end

    return 1 - res
end 
    

function entropie(y::AbstractVector)
    n = length(y)
    if n == 0
        return 0.0 
    end
    p0 = count(==(0),y)/n
    p1 = count(==(1),y)/n 
    if p0 == 0 && p1 !=0
        return -p1*log2(p1)
    elseif p0 !=0 && p1 == 0
        return -p0*log2(p0)
    elseif p0==0 && p1==0
        return 0.0
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

        gauche = y[A .<= seuil]#mutiple dispatching
        droite = y[A .> seuil]#multiple dispatching
        if isempty(gauche)|| isempty(droite)
            continue
        end

        gain = gini(y)-impureteGini(gauche,droite)

        if gain > best_gain
            best_gain = gain
            best_seuil = seuil

        end
    end
    
    return best_gain,best_seuil

end

function meilleure_colonne(X::AbstractMatrix,y::AbstractVector)
    best_gain = -Inf
    best_indice = nothing
    best_seuil = nothing
    for j in axes(X,2)# on parcours les variables explicatives
        A = X[:,j]
        gain,seuil = best_split(A,y)
        if gain>best_gain
            best_gain = gain
            best_indice = j
            best_seuil = seuil
        end

    end

    return best_gain,best_indice,best_seuil

end

function prediction_majoritaire(y::AbstractVector)
    n0 = count(==(0),y)
    n1 = count(==(1),y)
    return n1 > n0 ? 1  : 0
end

function construire_arbre(X,y,profondeur=0;
                          max_depth=5,
                          min_samples_split=10,
                          min_gain=0.0)

    prediction = prediction_majoritaire(y)

    # Condition d'arrêt 1 : une seule classe
    if length(unique(y)) == 1
        return Feuille(prediction)
    end

    # Condition d'arrêt  2 : profondeur maximale
    if profondeur >= max_depth
        return Feuille(prediction)
    end

    # Condition d'arrêt 3 : observations insuffisantes
    if length(y) < min_samples_split
        return Feuille(prediction)
    end

    # Recherche de la meilleure séparation
    gain, indice, seuil = meilleure_colonne(X,y)

    # Condition 4 : gain insuffisant
    if indice === nothing || gain <= min_gain
        return Feuille(prediction)
    end

    # Séparation des données
    masque_gauche = X[:,indice] .<= seuil
    masque_droite = .!masque_gauche

    Xg = X[masque_gauche,:]
    yg = y[masque_gauche]

    Xd = X[masque_droite,:]
    yd = y[masque_droite]

    # Construction récursive
    gauche = construire_arbre(
        Xg,yg,profondeur+1;
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_gain=min_gain
    )

    droite = construire_arbre(
        Xd,yd,profondeur+1;
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_gain=min_gain
    )

    return Noeud(indice,seuil,gauche,droite,prediction)
end

# construction de la fonction de Prédiction de l'arbre avec le multiple dispatching 

function predict(arbre::Feuille, x)
    return arbre.prediction
end

function predict(arbre::Noeud, x)

    if x[arbre.indice] <= arbre.seuil
        return predict(arbre.gauche,x)
    else
        return predict(arbre.droite,x)
    end
end

function predict(arbre::Arbre, X::AbstractMatrix)

    predictions = Vector{Int}(undef,size(X,1))

    for i in 1:size(X,1)
        predictions[i] = predict(arbre, X[i,:])
    end

    return predictions
end



"""Je vais essayer de diminuer la complexité des algorithmes (autant en espace
que en temps) en utilisant des techniques spécifiques à julia"""

# best split optimisé complexité O(n²) --> O(n*log(n)) : amélioration en temps de calcule


function best_splitO(A::AbstractVector{<:Real},
                    y::AbstractVector{<:Integer})

    n = length(y)

    if n < 2 || length(A) != n
        return -Inf, nothing
    end

    ordre = sortperm(A)
    n0 = count(==(0),y)
    n1 = count(==(1),y)

    # Impureté initiale
    impurite = 1-(n0/n)^2-(n1/n)^2

    # Effectifs à gauche
    n0g = 0
    n1g = 0

    best_gain = -Inf
    best_seuil = nothing

    for i in 1:n-1

        # Observation qui passe à gauche
        yi = y[ordre[i]]

        if yi == 0
            n0g += 1
            n0 -= 1
        elseif yi == 1
            n1g += 1
            n1 -= 1
        else
            throw(ArgumentError("Les classes doivent être 0 ou 1"))
        end

        # Ne pas séparer deux valeurs identiques
        a = A[ordre[i]]
        b = A[ordre[i+1]]

        if a == b
            continue
        end

        # Effectifs des deux groupes
        ng = i
        nd = n-i

        # Gini gauche
        gini_g = 1-(n0g/ng)^2-(n1g/ng)^2

        # Gini droit
        gini_d = 1-(n0/nd)^2-(n1/nd)^2

        # Impureté pondérée
        imp_apres = (ng/n)*gini_g + (nd/n)*gini_d

        # Gain
        gain = impurite-imp_apres

        if gain > best_gain
            best_gain = gain
            best_seuil = Float64(a)+(Float64(b)-Float64(a))/2
        end
    end

    return best_gain,best_seuil
end

# meilleure colonne optimisée : Diminution de l'allocation de la mémoire

function meilleure_colonneO(X::AbstractMatrix,
                           y::AbstractVector)

    best_gain = -Inf
    best_indice = nothing
    best_seuil = nothing

    for j in axes(X,2)

        A = view(X,:,j)

        gain,seuil = best_splitO(A,y)

        if gain > best_gain
            best_gain = gain
            best_indice = j
            best_seuil = seuil
        end
    end

    return best_gain,best_indice,best_seuil
end


function construire_arbreO(X,y,profondeur=0;
                          max_depth=5,
                          min_samples_split=10,
                          min_samples_leaf=5,
                          min_gain=0.0)

    isempty(y) && throw(ArgumentError("y ne doit pas être vide"))
    size(X,1) == length(y) ||
        throw(DimensionMismatch("X et y doivent avoir le même nombre de lignes"))

    prediction = prediction_majoritaire(y)

    # Une seule classe
    if all(==(y[1]),y)
        return Feuille(prediction)
    end

    # Profondeur maximale
    if profondeur >= max_depth
        return Feuille(prediction)
    end

    # Nombre d'observations insuffisant
    if length(y) < min_samples_split
        return Feuille(prediction)
    end

    gain,indice,seuil = meilleure_colonneO(X,y)

    # Gain insuffisant
    if indice === nothing || gain <= min_gain
        return Feuille(prediction)
    end

    masque_gauche = X[:,indice] .<= seuil
    masque_droite = .!masque_gauche

    # Vérifier la taille minimale des feuilles
    if count(masque_gauche) < min_samples_leaf ||
       count(masque_droite) < min_samples_leaf
        return Feuille(prediction)
    end

    Xg = X[masque_gauche,:]
    yg = y[masque_gauche]

    Xd = X[masque_droite,:]
    yd = y[masque_droite]

    gauche = construire_arbreO(
        Xg,yg,profondeur+1;
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_gain=min_gain
    )

    droite = construire_arbreO(
        Xd,yd,profondeur+1;
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_gain=min_gain
    )

    return Noeud(indice,seuil,gauche,droite,prediction)
end

