try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% code id=6780ad
import KaimonSlate

#%% code id=708198
using Revise
using ScratchML
using BenchmarkTools
using CSV,DataFrames

#%% md id=eaa189

#%% md id=68e2c1

#%% md id=f35cf6

#%% md id=e6d3ea

#%% md id=068245

#%% code id=d3e573
function gini(y::AbstractVector) # calcule l'indice de gini pour deux classe

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

#%% code id=141d55
df = CSV.read(
    "C:/Users/Admin/Desktop/ScratchML.jl/data/Titanic-Dataset.csv",
    DataFrame
)
print("fichier ouvert avec succès")

#%% code id=cb09e0
y = df.Survived
@btime gini(y)
@btime entropie(y)

#%% md id=15a7c2

#%% code id=22abd9

# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = c3c5841a-3637-41cc-bb4b-51c2ea95a9fc
#   format = 2
# ╚═╡
