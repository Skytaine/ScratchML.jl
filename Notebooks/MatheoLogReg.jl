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

#%% code id=590dc8
using MLJ, DataFrames, Random, Distributions
#on crée un dataset pour une régression logistique
#on crée 2 colonne explicative
n=25
age=floor.(Int,rand(Normal(45,16),n))
sexe=rand(["H","F"],n)
#on donne les coeff de la variable sexe
coef_sexe=Dict("H"=>0.9,"F"=>-0.2)
#sexe est standardisé (centré réduit) et on calcule a
a=-1 .- 0.3 .* (age .- 45)/4  .+ [coef_sexe[s] for s in sexe]

#la régression logistique doit nous donnée des valeurs proche du a_réel
a_réel=[-1,-0.3,0.9,-0.2]

p=1 ./ (1 .+ exp.(a))
class=Int.(rand(n) .> p)

df=DataFrame(age=age,sexe=sexe,class=class)

#%% code id=f76d36

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

# ╔═╡ Slate.env · notebook packages (auto-maintained — manage via the package panel)
#   GLMakie 0.13.15 e9467ef8-e4e7-5192-8a1a-b1aee30e663a
#   MLJ 0.23.3 add582a8-e3ab-11e8-2d5e-e98b27df1bc7
#   Plots 1.41.7 91a5bcdd-55d7-5caf-9e0b-520d859cae80
# ╚═╡
# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = 8e760128-3fa4-4eb7-9dee-473d59eaad61
# ╚═╡
