using HypothesisTests
using  CSV # ouverture du fichier CSV
using  DataFrames # Manipulation des données
using Statistics # Notions statistiques de base(moyenn,variance ...)
using StatsBase 
using StatsModels
using Plots
using GLM # Regression linéaire et logistique
using MLJ # Machine learning ?
using Flux # Réseau de neuronne


data = CSV.read("default_of_credit_card_clients.csv",DataFrame,delim = ";",header = 2)

function info_sur_donnes(df::DataFrame)
    println(size(df))
    println(describe(df))
end

#info_sur_donnes(data)

#= donc notre projet on cherche à estimater l'effet des varibales Xi
sur le défaut de payement Y d'un potentiel emprumter.
Y = {0,1}-->{oui,non} Donc Y est une varaible qualitative car le fait de 
changer les nombres par des mots ne change pas le sens de la variable =#

#= comme la variables est qualitative et que les autres sont quantitatives 
ou soit qualitatives , on va définir deux types de test pour comparer l'effet
chacune des autres variables sur Y=#

function test_quali_quanti(df, x, y)# test de Mannwhitney
    x_0 = df[df[!, y] .== 0, x]
    x_1 = df[df[!, y] .== 1, x]
    test = MannWhitneyUTest(x_0, x_1)
    return test.u, pvalue(test)
end

function test_quali_quali(x, y)

    categories_x = unique(x)
    categories_y = unique(y)

    table = [
        sum((x .== xi) .& (y .== yi))
        for xi in categories_x, yi in categories_y
    ]

    test = ChisqTest(table)

    return test.stat, pvalue(test)
end

function visualisation_quali_quali(df, x, y) # Diagramme en barre
    categories_x = unique(df[!, x])
    categories_y = unique(df[!, y])

    effectifs = [
        sum((df[!, x] .== xi) .& (df[!, y] .== yi))
        for xi in categories_x, yi in categories_y
    ]

    bar(
        string.(categories_x),
        effectifs,
        group = string.(categories_y),
        xlabel = string(x),
        ylabel = "Effectif",
        title = "$(x) selon $(y)"
    )
end

function visualisation_quali_quanti(df, x, y) # Boxplot
    categories = unique(df[!, x])

    valeurs = [
        df[df[!, x] .== categorie, y]
        for categorie in categories
    ]

    boxplot(
        valeurs,
        labels = string.(categories),
        xlabel = string(x),
        ylabel = string(y),
        title = "$(y) selon $(x)"
    )
end
#= Hypothèse des tests: 
Hypothèse nulles : Les distributions sont sont indépendantes entres eux -->
la distribution de Y ne dépend pas de X
Hypothèse Alternatif : La distribution de Y ne dépend pas de X =#

# Fonction faisant automatiquement les analyses

function analyse_quali_quanti(df, x, y)

    println("-----------------------------------")
    println("Variable : ", x)
    println("Cible : ", y)

    visualisation_quali_quanti(df, y, x)

    u, p = test_quali_quanti(df, x, y)

    println("Statistique U = ", u)
    println("p-value = ", p)

    if p < 0.05
        println(" Association statistiquement significative")
    else
        println("Pas d'association statistiquement significative")
    end
end

function analyse_quali_quali(df, x, y)

    println("-----------------------")
    println("Variable : ", x)
    println("Cible : ", y)

    visualisation_quali_quali(df, x, y)

    chi2, p = test_quali_quali(df[!, x], df[!, y])

    println("Statistique chi-2 = ",chi2)
    println("p-value = ", p)

    if p < 0.05
        println("Association statistiquement significative")
    else
        println("Pas d'association statistiquement significative")
    end
end