using Projet_Machine_Learning
using DataFrames
using Distributions
using CSV
using RDatasets
#df=DataFrame(CSV.File("Data/default_of_credit_card_clients.csv",header=1))
iris= dataset("datasets", "iris")
unique(iris[!,"Species"])

Projet_Machine_Learning.LogReg()