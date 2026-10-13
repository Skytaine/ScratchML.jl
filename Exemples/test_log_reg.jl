#Notre package
using ScratchML
#Package pour comparer
using MLJ
#Pour utiliser des dataframe et csv
using DataFrames
using CSV
#Pour avoir des dataframes exemples usuelles
using RDatasets

iris= dataset("datasets", "iris")
typeof(iris)
