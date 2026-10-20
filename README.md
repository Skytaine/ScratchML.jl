# ScratchML
## Objectif
Coder de zéro un package de machine learning similaire à [MLJ.jl](https://github.com/JuliaAI/MLJ.jl) et comparer les modèles codés avec ceux fournis par le `MLJ.jl`.

## Structure

### Types

Ce package inclut un type abstrait `AbstractModel`. Les modèles sont des types concret qui contiennent des paramètres et hérite du type `AbstractModel`.

### Fonctions

Les 2 fonctions principales sont :

1) Une fonction pour entrainer un modèle supervisé ou non supervisé sur des données.

| Supervisé | Non supervisé |
|---|---|
| `fit!(model,donnée,cible)` | `fit!(model,donnée)` |

2) Une fonction de prédiction  `predict(model,donnée)`
