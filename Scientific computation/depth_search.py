import networkx as nx
G=nx.Graph()
edges=[[1,2],[1,3],[1,2],[2,4],[3,4],[5,6]]
G.add_edges_from(edges)
s=1
z=[-1000 for i in range(G.number_of_nodes())]
L=[list(G.nodes()),z]
#L=[nodes,z,d]
D=z.copy() #list of distances
S=[]
S.append(s)
L[1][s-1]=1
D[s-1] = 0
while len(S)>0:
    v=S.pop()
    print("Removing node %d" %(v))
   
    for i in G.adj[v]:
        if L[1][i-1]==-1000:
            L[1][i-1]=1
            S.append(i)
            D[i-1] = D[v-1] + 1
            #L[2][v-1] = L[2][n-1] +1
    print("Stack=",S)
print(L)
print(D)