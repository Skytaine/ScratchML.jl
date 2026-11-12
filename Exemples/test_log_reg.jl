#Notre package
using ScratchML
#Package pour comparer
using MLJ
#Pour utiliser des dataframe et csv
using DataFrames
using MLJLinearModels


#On créer des données pour une régression logistique avec n lignes et p colonnes.
n=200
p=2


X, y = make_blobs(n, p; centers=2)

X_mat=Matrix(DataFrame(X))
y_mat=Vector(y)

model=LogReg()


fit!(model,X_mat,y_mat .- 1)

y_pred=ScratchML.predict(model,X_mat)

#Précision
1 - sum(abs.( y_mat .-1 - y_pred))/length(y_mat)


using MLJLinearModels

modelMLJ=LogisticClassifier()
mach=MLJ.fit!(machine(modelMLJ,X,y))
y_pred_mlj=MLJ.predict_mode(mach,X)

accuracy(y_pred_mlj,y)