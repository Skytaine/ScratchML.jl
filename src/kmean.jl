
function distance(a,b)
    return sum(abs2, a.-b)
end


struct InitAlea <: AbstractModel end 
struct InitChoice <: AbstractModel end



function init_flags(::InitAlea, X, k)  # on pose les centroides de manières aleatoires
    n = size(X,2)
    flags = [rand(1:n)]
    while length(flags) < k
        i = rand(1:n)
        if !(i in flags)
            push!(flags, i)
        end
    end
    return X[:,flags]
end


function init_flags(::InitChoice , X , k) # on pose le premier centroides de manières aleatoires puis on place les suivants en ajoutant un poids en fct de la distance pour poser les drapeaux suivants
    n=size(X,2)
    flags=[rand(1:n)]
    while length(flags)<k
        d2=[minimum(distance(X[:,i],X[:,j]) for j in flags) for i in 1:n]
        r=rand()*sum(d2)
        c=cumsum(d2)
        for i in 1:n
            if c[i] >= r 
                push!(flags,i)
                break
            end
        end
    end
    return X[:,flags]
end



function init_flagsbis(::InitAlea , X , k)
    flags=randperm(size(X,2)) #randperm permet de melanger directement la liste
    return X[:,flags[1:k]]
end




function assigned(X,C)
    n,k = size(X,2) , size(C,2)
    d2 = zeros(n, k)
    for i in 1:n
        for j in 1:k
            d2[i, j] = distance(X[:, i], C[:, j])
        end
    end
    closer=[argmin(d2[i,:] ) for i in 1:n]  #argmin donne la position de la plus petite valeur
    total_distance=sum(d2[i,closer[i]] for i in 1:n)
    return closer , total_distance
end



function repo(X, closer, k)
    C = zeros(size(X, 1), k)
    for j in 1:k
        team = findall(closer .== j) #findall cherche où se trouvent les cases « vrai » dans une liste, et rend leurs numéros (position) sous une liste. et closer .== et pas == car .== parcourt pour chaque valeur de la liste et == demande si la liste entière est égale
        if length(team) == 0 # si personne associé au centroides on le positionne au hasard
            C[:, j] = X[:, rand(1:size(X, 2))]
        else
            C[:, j] =(mean(X[:, team])) 
        end
    end
    return C
end
