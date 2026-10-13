try; import KaimonSlate; catch; error("This is a Kaimon Slate notebook — running it as plain Julia needs the KaimonSlate runtime in this environment. Add it with `import Pkg; Pkg.add(\"KaimonSlate\")`, or open it in Kaimon Slate."); end; KaimonSlate.standalone!(@__MODULE__; dir=@__DIR__)

#%% md id=intro
@md"""
# Kmean
"""

#%% md id=f44cc3
@md raw"""
## Ce que l'on veut

Mesurer à quel point deux points `a` et `b` sont éloignés :

$$d(a, b) = \sum_{r} (a_r - b_r)^2$$

C'est la **distance euclidienne au carré**. On garde le carré (pas de racine) car, pour savoir quel point est le plus proche, le classement est le même : si $d_1 < d_2$ alors $\sqrt{d_1} < \sqrt{d_2}$.
"""

# ╔═╡ Slate.config · per-notebook settings (Settings panel)
#   docid = 005ef319-bf6a-4777-8f08-776c381f816c
# ╚═╡
