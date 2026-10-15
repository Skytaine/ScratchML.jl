
function distance(a,b)
    return sum(abs2, a.-b)
end


struct InitAlea <: AbstractModel end 
struct InitChoice <: AbstractModel end



function init_flags(::InitAlea, X, k)
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


function init_flags(::InitChoice , X , k)
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
    flags=randperm(size(X,2))
    return X[:,flags[1:k]]
end




