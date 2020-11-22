#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:37:45 2020

@author: andreeateodora
"""
import networkx as nx
G=nx.Graph()
edges=[[1,2],[1,3],[1,2],[2,4],[3,4],[5,6]]
G.add_edges_from(edges)
s=1
z=[-1000 for i in range(G.number_of_nodes())]
L=[list(G.nodes()),z]
#L=[nodes,z,d]
D=z.copy() #list of distances
Q=[]
Q.append(s)
L[1][s-1]=1
D[s-1] = 0
while len(Q)>0:
    v=Q.pop(0)
    print("Removing node %d" %(v))
    print("Queue=",Q)
    for i in G.adj[v]:
        if L[1][i-1]==-1000:
            L[1][i-1]=1
            Q.append(i)
            D[i-1] = D[v-1] + 1
            #L[2][v-1] = L[2][n-1] +1
print(L)
print(D)
