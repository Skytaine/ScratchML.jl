try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# Kmean
"""

#%% md id=contexte
@md raw"""
## 1. Contexte et objectif
 
K-Means est une méthode de **classification non supervisée** (*clustering*) : à partir de $n$ observations $x_1, \dots, x_n \in \mathbb{R}^d$, sans variable réponse $y$, on cherche une partition en $k$ groupes $S_1, \dots, S_k$ de points proches les uns des autres.
 
Le critère est l'**inertie intra-classe** (somme des carrés intra-classe) :
 
$$J(S, C) = \sum_{j=1}^{k} \sum_{x_i \in S_j} \| x_i - c_j \|^2$$
 
où $c_j \in \mathbb{R}^d$ est le **centroïde** du groupe $S_j$. Minimiser $J$ sur toutes les partitions est un problème combinatoire NP-difficile dans le cas général : on se contente d'une heuristique itérative qui converge vers un minimum local.
"""

#%% md id=conception
@md raw"""
## Conception de l'implémentation
 
**Conventions.** La matrice de données $X \in \mathbb{R}^{d \times n}$ contient **une observation par colonne** : Julia stocke les matrices par colonnes, ce qui rend l'accès à un point contigu en mémoire. C'est aussi la convention de `Clustering.jl`.
 
| Notation | Code | Contenu |
|---|---|---|
| $X$ | `X` | données, matrice $d \times n$ |
| $C$ | `flags` | centroïdes, matrice $d \times k$ |
| étiquettes de $S$ | `closer` | numéro du groupe de chaque observation |
| $J$ | `total_distance` | inertie |
 
**Architecture.** Les stratégies d'initialisation sont des **types** (`InitAlea`, `InitChoice`, sous-types de `AbstractInit`), et la fonction `init_flags` possède **une méthode par type**. Julia sélectionne la méthode par *multiple dispatch* : ajouter une stratégie revient à définir un type et une méthode, sans modifier `fit!`. De même, `fit!` et `predict` sont définies sur le type `KMeansModel`, ce qui permet à d'autres modèles de partager ces noms (une régression logistique, par exemple).
 
| Fonction | Rôle |
|---|---|
| `distance` | distance euclidienne au carré |
| `init_flags` | initialisation des centroïdes (une méthode par stratégie) |
| `assigned` | étape d'affectation et calcul de l'inertie |
| `repo` | étape de mise à jour des centroïdes |
| `KMeansModel` | structure de données : paramètres et résultats |
| `fit!` | boucle principale de l'algorithme |
| `predict` | affectation de nouvelles observations |
"""

#%% md id=distance_txt
@md raw"""
### distance
 
**Spécification.** Entrées : deux vecteurs `a` et `b` de même longueur. Sortie : $\| a - b \|^2 = \sum_r (a_r - b_r)^2$.
 
**Implémentation.** `a .- b` effectue la soustraction terme à terme (*broadcasting*), `abs2` élève au carré, `sum` additionne.
 
**Remarque.** On conserve le carré : l'`argmin` des distances est inchangé par la racine carrée (fonction croissante), et l'inertie est elle-même définie avec des carrés.
"""

#%% md id=f44cc3
@md raw"""
## Ce que l'on veut

Mesurer à quel point deux points `a` et `b` sont éloignés :

$$d(a, b) = \sum_{i} (a_i - b_i)^2$$

C'est la **distance euclidienne au carré**. On garde le carré (pas de racine) car, pour savoir quel point est le plus proche, le classement est le même : si $d_1 < d_2$ alors $\sqrt{d_1} < \sqrt{d_2}$.
"""

# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = 005ef319-bf6a-4777-8f08-776c381f816c
# ╚═╡
