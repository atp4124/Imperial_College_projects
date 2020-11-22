#shortest distance between the source and all of the other nodes

import networkx as nx
e=[[1,2,3],[1,5,2],[2,5,1],[2,3,1],[3,4,2],[4,5,5],[4,6,1]]
G=nx.Graph()
G.add_weighted_edges_from(e)

Udict={} #priority queue
Udict[5]=0 #source initialisation
Edict={}
while len(Udict):
    #find highest priority node in Udict and move it to Edict
    dmin=200000
    for k,v in Udict.items():
        if v<dmin:
            dmin = v
            kmin = k
    Edict[kmin] = Udict.pop(kmin)
    print("removing node %d with distance %d" %(kmin,dmin))
    #iterate through the neighbours of removed nodes
    for n,i,w  in G.edges(kmin,data='weight'):
        if i in Edict:
            pass
        elif i in Udict:
            dnew = dmin + w    
            if dnew<Udict[i]:
                Udict[i]=dnew
        else:
            dnew = dmin + w
            Udict[i] = dnew
        print("Udict",Udict)
        print("Edict",Edict)
         #check if the neighbour is in Edict, Udict