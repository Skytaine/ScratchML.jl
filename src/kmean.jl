function distance(::EuclidienneCarree, a::AbstractVector, b::AbstractVector)
    s = 0.0
    for r in 1:lenght(a)
        d = a[r] - b[r]
        s += d * d
    end
    return s
end