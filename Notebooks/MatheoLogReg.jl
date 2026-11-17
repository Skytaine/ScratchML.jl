try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md raw"""
# Régression Logistique

On veut expliquer une variable binaire Y=0 ou 1 par $n$ variables explicatives $(X_1,\dots,X_n)$.   
   
Pour ça on note 
$$p=P(Y=1|X)$$
et on introduit le LOGIT de p
$$ LOGIT (p)=ln(\frac{p}{1-p}). $$

Pour une réalistation $x=(x_1,\dots,x_n)$  
On approche le LOGIT par 
$$  ln(\frac{p}{1-p})=a_0+a_1x_1+\dots+a_nx_n$$
et en inversant on obtient une formule pour p,
$$ \sigma(x)=p=\frac{1}{1+e^{-(a_0+a_1x_1+\dots+a_nx_n)}}. $$
On veut estimer le vecteur $a=(a_0,\dots,a_n)$ par le maximum de vraisemblance.


"""

#%% code id=e3fd92 hidecode
using MLJ, MLJTransforms
using Random, Distributions
using DataFrames, CSV
#on crée un dataset pour une régression logistique
#on crée 2 colonne explicative
n=20
age=floor.(Int,rand(Normal(45,16),n))
sexe=rand(["H","F"],n)
#on donne les coeff de la variable sexe
coef_sexe=Dict("H"=>0.9,"F"=>-0.2)
#sexe est standardisé (centré réduit) et on calcule a
a=-1 .- 0.3 .* (age .- 45) ./4  .+ [coef_sexe[s] for s in sexe]

#la régression logistique doit nous donnée des valeurs proche du a_réel
# pour avoir autant de chaque classe on veut mean(a)=log(0.5)
a_réel=[-1,-0.3,-0.2,0.9] 

p=1 ./ (1 .+ exp.(-a))
class=Int.(0.5 .> p)

df=DataFrame(age=age,sexe=sexe,class=class)


#On utilise cet version préparé pour MLJ et ScratchML
df2=MLJ.transform(MLJ.fit!(machine(OneHotEncoder(),coerce(df,:sexe=>Multiclass,:age=>Count,:class=>Count))))
df2.age=(age .- 45)./4
typeof(df2)
#df2[!,[:class]] pour extraire une colonne

#%% md id=455c05
@md"""
# Visualisation
On a la fonction sigmoïde en bleu et notre seuil en rouge.   
Les points représente une prédiction de la classe d'une ligne :  
Au dessus du seuil -> classe 1   
En dessous -> classe 0   
Les points coïncide lorsque les coefficent a_0, a_a, a_F, a_H sont au valeurs de a_réel  
$$a_{réel}=[-1,-0.3,-0.2,0.9]. $$
"""

#%% code id=f4782b
#slider des 4 paramètres
@bind a_0 Slider(range(-2,step=0.1,length=41),default=-1)
@bind a_a Slider(range(-2,step=0.1,length=41),default=-0.3)
@bind a_F Slider(range(-2,step=0.1,length=41),default=-0.2)
@bind a_H Slider(range(-2,step=0.1,length=41),default=0.9)

a_test=[a_0,a_a,a_F,a_H]
a_pred=Matrix(hcat(ones(n),df2)[:,Not(:class)])*a_test
p_pred=1 ./ (1 .+ exp.(-a_pred))
print("Accuracy = ", sum(Int.(0.5 .> p_pred) .== class)/n*100 , "%")

echart(series(:line, -10:10, 1 ./ (1 .+ exp.(-(-10:10))),showSymbol=false),series(:scatter, a, p),series(:line,-10:10,0.5 .+ 0* (-10:10),showSymbol=false),lineStyle = (type = "dashed"),series(:scatter, a_pred, p_pred))

#%% md id=60565f
@md"""
# Plan du programme
On s'inspire du [code python d'une régression logistique](https://www.kaggle.com/code/fareselmenshawii/logistic-regression-from-scratch#Model).   
Fonction :   
1)LogReg(a) regression logistique avec vecteur a fixée   
2)Fit(DataFrame ou un duo var explicative et var a expliquer,LogReg)->fonction LogReg avec vecteur a optimisé par max vraisemblance    
3)Predict(DataFrame ou vecteur,LogReg) a chaque ligne d'un dataframe donne le résultat de la regression logistique entrainée.  


# LogReg
mutable struct LogReg contenant un vecteur réel

# Fit(Data,LogReg)
Le Data contient les variables explicative X et la variable a expliquer Y.  
Calcul du maximum de vraisemblance (descente de gradient)
"""

#%% code id=5440e6
using DataFrames
using MLJ
using ScratchML

#On utilise MLJ pour faire un dataset artificiel pour la régression logistique
n=200
p=2

centre=[-10 -10; 10 10]

X_mat, y = make_blobs(n, p; centers=centre, as_table=false)


model=LogReg()

ScratchML.fit!(model,X_mat,y .- 1)

y_pred=ScratchML.predict(model,X_mat)

#Précision
println("\nPrécision =",accuracy( y .-1 , y_pred)*100,"%")

println("\nOn a un grande précision quand les centres sont éloignés")

index=Bool.(y .- 1)

echart(series(:scatter,X_mat[index,1],X_mat[index,2]),series(:scatter,X_mat[.!index,1],X_mat[.!index,2]))

#%% code id=591bcc
using DataFrames
using MLJ
using ScratchML

#On utilise MLJ pour faire un dataset artificiel pour la régression logistique
n=2000
p=2

centre=[-1 0; 0 1]

X_mat, y = make_blobs(n, p; centers=centre, as_table=false)


model=LogReg()

ScratchML.fit!(model,X_mat,y .- 1)

y_pred=ScratchML.predict(model,X_mat)

#Précision
println("\nPrécision =",accuracy( y .-1 , y_pred)*100,"%")

println("\nOn a une précision plus faible les ensembles se superposent")

index=Bool.(y .- 1)

echart(series(:scatter,X_mat[index,1],X_mat[index,2]),series(:scatter,X_mat[.!index,1],X_mat[.!index,2]))

#%% code id=b7571c hidecode
using DataFrames
using MLJ
using ScratchML

#On utilise MLJ pour faire un dataset artificiel pour la régression logistique
n=2000

X_mat, y = make_circles(n, as_table=false)

model=LogReg()

ScratchML.fit!(model,X_mat,y )

y_pred=ScratchML.predict(model,X_mat)

#Précision
println("\nPrécision =",accuracy( y  , y_pred)*100,"%")

println("\nOn a une précision très faible pour prédire des classes qui ne sont pas linéaire")

index=Bool.(y)
echart(series(:scatter,X_mat[index,1],X_mat[index,2]),series(:scatter,X_mat[.!index,1],X_mat[.!index,2]))

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   CSV 0.10.17 336ed68f-0bac-5ca0-87d4-7b16caf5d00b
#   MLDatasets 0.7.22 eb30cadb-4394-5ae3-aed4-317e484a6458
#   MLJTransforms 0.1.8 23777cdb-d90c-4eb0-a694-7c2b83d5c1d6
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = 9cde95f1-4d41-4a35-9bd6-6acad98e5462
# ╚═╡
