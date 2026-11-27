module ScratchML

using Random, Statistics
using DataFrames
import StatsAPI
import StatsAPI: fit!, predict

# Fonctions exportées
export distance, init_flags, init_flagsbis, assigned, mouv
export InitAlea, InitChoice
export logvrai, gradient, ajout_col_1, sigm, LogReg
export Arbre, Feuille, Noeud
export gini, best_split, construire_arbre
export best_splitO, construire_arbreO
export meilleure_colonne,meilleure_colonneO
export fit!, predict

abstract type AbstractModel end

# Inclusion des fichiers
include("logreg.jl")
include("kmean.jl")
include("ArbreDecision.jl")

end
