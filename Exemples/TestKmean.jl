using ScratchML
X=[1.0 15.0 45.0 16.0 94.0]
init_flags(InitAlea(),X,2)
init_flags(InitChoice(),X,2)
init_flagsbis(InitAlea(),X,2)

X1 = [1.0 2.0 3.0 10.0 11.0 12.0]
X2 = [0.0 0.0 10.0 10.0;
      0.0 1.0 10.0 11.0]  
#fct distance

distance([0.0, 0.0], [3.0, 4.0])     # 25.0
distance([1.0, 1.0], [1.0, 1.0])     # 0.0
distance([1.0], [4.0])               # 9.0


#fct assigned

closer, total = assigned(X1, [0.0 12.0])
closer         # [1, 1, 1, 2, 2, 2]
total          # 19.0   (1 + 4 + 9 + 4 + 1 + 0)

assigned(X2, [0.0 10.0; 0.0 10.0])    # ([1, 1, 2, 2], 2.0)


#fct repo
# X2
# k=3
#  C = zeros(size(X2, 1), k)

# ScratchML.Statistics.mean(X2[:, 1:2],dims=2)

repo(X1, [1, 1, 1, 2, 2, 2], 2)        # [2.0 11.0]
repo(X2, [1, 1, 2, 2], 2)             # [0.0 10.0; 0.5 10.5]

C = repo(X1, [1, 1, 1, 1, 1, 1], 2)
C[1, 1]                # 6.5 (la moyenne des 6)
C[1, 2] in X1           # true (drapeau replacé sur un enfant)
