using ScratchML
using BenchmarkTools
using MLJ
using CSV
using DataFrames

# Evaluation des optimisations en temps et en mémoire

df = CSV.read("data/Titanic-Dataset.csv",DataFrame)
println(df)

names(df)
y = df.Survived
A = df.Parch
@btime best_split($A,$y)
@btime best_splitO($A,$y)
print(A)