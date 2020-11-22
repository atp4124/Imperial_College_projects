"""Scientific Computation Project 2, part 2
Your CID here:01365348
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy as sp



def rwgraph(G,i0=0,M=100,Nt=100):
    """ Question 2.1
    Simulate M Nt-step random walks on input graph, G, with all
    walkers starting at node i0
    Input:
        G: An undirected, unweighted NetworkX graph
        i0: intial node for all walks
        M: Number of walks
        Nt: Number of steps per walk
    Output: X: M x Nt+1 array containing the simulated trajectories
    """

    #matrix for X which stores the ndoes
    X = np.zeros((M,Nt+1),dtype=np.int)
    #set the starting node to equal i0
    X[:,0]=i0
    #iterate through M simulations
    for i in range(M):
        v=i0
        #iterate through Nt steps
        for k in range(Nt):
            #get the list of neighbours of v
            adj=list(G.adj[v])
            #choose a random neighbour
            v=np.random.choice(adj)
            #add the neighbour to matrix X
            X[i][k+1]=v
    return X
 

    
def rwgraph_analyze1(graph):
    """Analyze simulated random walks on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    #values to go through for M
    M=[100,200,300,500,1000,6000]
    #values to go through for Nt
    Nt=[100,250,350,550,1500,2500]
    #calculate the maximum node
    maxnode=max(graph.degree, key=lambda x: x[1])[0]
    #plot for each value of M and Nt
    (fig,ax)=plt.subplots(2,3)
    nodeseq=rwgraph(graph,i0=maxnode,M=M[0],Nt=Nt[0])
    ax[0,0].hist(nodeseq[:,Nt[0]],bins=graph.number_of_nodes())
    ax[0,0].set_xlabel("Node number",fontsize=15)
    ax[0,0].set_ylabel("Node frequency",fontsize=15)
    ax[0,0].set_title("M=100, Nt=100",fontsize=15)
    nodeseq=rwgraph(graph,i0=maxnode,M=M[1],Nt=Nt[1])
    ax[0,1].hist(nodeseq[:,Nt[1]],bins=graph.number_of_nodes())
    ax[0,1].set_xlabel("Node number",fontsize=15)
    ax[0,1].set_ylabel("Node frequency",fontsize=15)
    ax[0,1].set_title("M=200,Nt=250",fontsize=15)
    nodeseq=rwgraph(graph,i0=maxnode,M=M[2],Nt=Nt[2])
    ax[0,2].hist(nodeseq[:,Nt[2]],bins=graph.number_of_nodes())
    ax[0,2].set_xlabel("Node number",fontsize=15)
    ax[0,2].set_ylabel("Node frequency",fontsize=15)
    ax[0,2].set_title("M=300,Nt=350",fontsize=15)
    nodeseq=rwgraph(graph,i0=maxnode,M=M[3],Nt=Nt[3],)
    ax[1,0].hist(nodeseq[:,Nt[3]],bins=graph.number_of_nodes())
    ax[1,0].set_xlabel("Node number",fontsize=15)
    ax[1,0].set_ylabel("Node frequency",fontsize=20)
    ax[1,0].set_title("M=500,Nt=550",fontsize=15)
    nodeseq=rwgraph(graph,i0=maxnode,M=M[4],Nt=Nt[4])
    ax[1,1].hist(nodeseq[:,Nt[4]],bins=graph.number_of_nodes())
    ax[1,1].set_xlabel("Node number",fontsize=15)
    ax[1,1].set_ylabel("Node frequency",fontsize=15)
    ax[1,1].set_title("M=1000,Nt=1500",fontsize=15)
    nodeseq=rwgraph(graph,i0=maxnode,M=M[5],Nt=Nt[5])
    ax[1,2].hist(nodeseq[:,Nt[5]],bins=graph.number_of_nodes())
    ax[1,2].set_xlabel("Node number",fontsize=15)
    ax[1,2].set_ylabel("Node frequency",fontsize=15)
    ax[1,2].set_title("M=6000,Nt=2500",fontsize=15)
    

def rwgraph_analyze2(graph):
    """Analyze simulated random walks on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    #calculate the maximum node
    maxnode=max(graph.degree, key=lambda x: x[1])[0]
    #calculate a 15 3000-steps random walk
    nodeseq=rwgraph(graph,i0=maxnode, M=15,Nt=3000)
    #get the list of degrees for the 4th, 7th and 15th iteration
    list_degrees1=sorted([graph.degree[i] for i in nodeseq[3,:]])
    list_degrees2=sorted([graph.degree[i] for i in nodeseq[7,:]])
    list_degrees3=sorted([graph.degree[i] for i in nodeseq[14,:]])
    #plot the degrees on the x axis and the frequency on the y axis
    (fig,ax)=plt.subplots(1,3)
    ax[0].hist(list_degrees1,bins=sorted(np.array(list(set(list_degrees1)))))
    ax[0].set_xlabel("Degree",fontsize=20)
    ax[0].set_ylabel("Frequency",fontsize=20)
    ax[0].set_title("4th simulation",fontsize=20)
    ax[1].hist(list_degrees2,bins=sorted(np.array(list(set(list_degrees2)))))
    ax[1].set_xlabel("Degree",fontsize=20)
    ax[1].set_ylabel("Frequency",fontsize=20)
    ax[1].set_title("8th simulation",fontsize=20)
    ax[2].hist(list_degrees3,bins=sorted(np.array(list(set(list_degrees3)))))
    ax[2].set_xlabel("Degree",fontsize=20)
    ax[2].set_ylabel("Frequency",fontsize=20)
    ax[2].set_title("15th simulation",fontsize=20)


def linear_diffusion(G,tf=5,Nt=2000,i0=0.1):
    '''
    Calculate the solution for the linear diffusion
    '''
    #number of nodes of the graph
    N = G.number_of_nodes()
    #list of degrees for each node
    degree=[degree for node,degree in G.degree()]
    #calculate the maximum degree
    maxnode=max(G.degree, key=lambda x: x[1])[0]
    #get the adjacency matrix of the graph
    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype = np.float64)
    #diagonal matrix with diagonal entries as the degree of the node
    D = np.diag(degree)
    #laplacian matrix
    L=D-A
    #initial condition vector
    ini_cond=np.zeros(N)
    #set the magnitude for the initial node
    ini_cond[maxnode]=i0
    #scaled laplacian matrix
    L_s=np.identity(N)-np.matmul(np.linalg.inv(D),A)
    #transpose of the scaled laplacian
    L_s_t=np.transpose(L_s)
    #calculate the solution at time =5
    expL=np.matmul(sp.linalg.expm(-5*L),ini_cond)
    expL_s=np.matmul(sp.linalg.expm(-5*L_s),ini_cond)
    explL_s_t=np.matmul(sp.linalg.expm(-5*L_s_t),ini_cond)
    solL=expL
    solL_s=expL_s
    solL_s_t=explL_s_t
    
    #plot the solution as a function of node
    fig1=plt.figure(figsize=(20,10))
   
    plt.title("Solution for Laplacian L with t=5",fontsize=20)
    plt.xlabel("Node label",fontsize=20)
    plt.ylabel("Solution",fontsize=20)
    plt.plot(range(N),solL)
    fig1.show()
    
    fig2=plt.figure(figsize=(20,10))
    plt.plot(range(N),solL_s)
    plt.title("Solution for Scaled Laplacian L with t=5",fontsize=20)
    plt.xlabel("Node label",fontsize=20)
    plt.ylabel("Solution",fontsize=20)
    fig2.show()
    
    fig3=plt.figure(figsize=(20,10))
    plt.plot(range(N),solL_s_t)
    plt.title("Solution for Transposed Scaled Laplacian L with t=5",fontsize=20)
    plt.xlabel("Node label",fontsize=20)
    plt.ylabel("Solution",fontsize=20)
    fig3.show()

def modelA(G,x=0,i0=0.1,beta=1.0,gamma=1.0,tf=5,Nt=1000):
    """
    Question 2.2
    Simulate model A

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    beta,gamma: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: N x Nt+1 Array containing i across network nodes at
                each time step.
    """
    #number of nodes
    N = G.number_of_nodes()
    #solution i array
    iarray = np.zeros((N,Nt+1))
    #time array
    tarray = np.linspace(0,tf,Nt+1)
    #adjacency matrix of the graph
    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype = np.float64)
    #initial conditions of the graph
    initial_cond=np.zeros(N)
    initial_cond[x]=i0
    
    
    def RHSA(y,t):
        """Compute RHS of modelA at time t
        input: y should be a size N array
        output: dy, also a size N array corresponding to dy/dt

        Discussion: add discussion here
        """
     
        dy=-beta*y[0:N]+gamma*(np.ones(N)-y[0:N])*np.transpose(np.matmul(A,y[0:N]))
        
        return dy

    #final solution of the differential equation
    iarray=np.transpose(odeint(RHSA,initial_cond,tarray))
    
    return iarray




def modelB(G,x=0,i0=0.2,i1=0.2,tf=5,Nt=1000,alpha=1.0):
    """
    Question 2.2
    Simulate model A

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    alpha: model parameters
    tf,2*Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: 2*N x 2*Nt+1 Array containing i across network nodes at
                each time step.
    """
    #number of nodes
    N = G.number_of_nodes()
    #solution i array
    iarray = np.zeros((2*N,Nt+1))
    #time array
    tarray = np.linspace(0,tf,Nt+1)
    #degree list for each node
    degree=[degree for node,degree in G.degree()]
    #adjacency matrix
    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype = np.float64)
    #diagonal matrix with diagonal entriea as the degree of the node
    D = np.diag(degree)
    #laplacian matrix
    L=D-A
    #initial conditions
    initial_cond=np.zeros((N,))
    initial_cond1=np.zeros((N,))
    initial_cond[x]=i0
    initial_cond1[x]=i1
    initial_cond=np.concatenate((initial_cond,initial_cond1))
    
    def RHSB(y,t):
        """Compute RHS of modelA at time t
        input: y should be a size N array
        output: dy, also a size N array corresponding to dy/dt

        Discussion: add discussion here
        """
        dy=np.concatenate((alpha*np.matmul(L,y[N:2*N]),y[0:N]))
        
        return dy


    iarray=np.transpose(odeint(RHSB,initial_cond,tarray))
    
    return iarray

def transport(graph,Nt=2000,tf=5,alpha=-0.01,beta=0.5,gamma=0.1):
    """Analyze transport processes (model A, model B, linear diffusion)
    on Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    #maximum degree node
    maxnode=max(graph.degree, key=lambda x: x[1])[0]
    #solution from model A and model B
    iarray=modelA(graph,x=maxnode,beta=beta,gamma=gamma,Nt=Nt,tf=tf)
    iarrayB=modelB(graph,x=maxnode,alpha=alpha,Nt=Nt,tf=tf)
    #plot the solutions
    (fig,ax)=plt.subplots(1,2)
    for i in range(graph.number_of_nodes()):
        ax[0].plot(np.linspace(0,tf,Nt+1),iarray[i,:])
        ax[0].set_xlabel("Time step",fontsize=20)
        ax[0].set_ylabel("Infected fraction",fontsize=20)
        ax[0].set_title("Model A",fontsize=20)
        
    for i in range(graph.number_of_nodes()):
        ax[1].plot(np.linspace(0,tf,Nt+1),iarrayB[i,:])
        ax[1].set_xlabel("Time step",fontsize=20)
        ax[1].set_ylabel("Infected fraction",fontsize=20)
        ax[1].set_title("Model B",fontsize=20)
    #modify as needed

def transport3(graph,Nt=2000,tf=20):
    """Analyze transport processes (model A, model B, linear diffusion)
    on Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    #maximum degree node
    maxnode=max(graph.degree, key=lambda x: x[1])[0]
    #solution of model B
    iarray=modelB(graph,x=maxnode,alpha=-0.01, Nt=Nt,tf=tf)
    #plot the solution of the last node
    plt.plot(range(graph.number_of_nodes()),iarray[0:100,Nt])
    plt.xlabel("Node number",fontsize=20)
    plt.ylabel("Infected fraction at the last itearation",fontsize=20)
        
        
def transport2(graph,Nt=2000,tf=20):
    #maximum degree node
    maxnode=max(graph.degree, key=lambda x: x[1])[0]
    #solution of model A
    iarray=modelA(graph,x=maxnode,beta=0.5,gamma=0.1,Nt=Nt)
    #plot the solution of the last node
    plt.plot(range(graph.number_of_nodes()),iarray[:,Nt])
    plt.xlabel("Node number",fontsize=20)
    plt.ylabel("Infected fraction at the last itearation",fontsize=20)
 

    
    return iarray
if __name__=='__main__':
   
    
    H=nx.barabasi_albert_graph(2000,4)
    H1=nx.barabasi_albert_graph(100,8)
    rwgraph_analyze1(H)
    rwgraph_analyze2(H)
    linear_diffusion(H)
    transport(H)
    transport(H,tf=20)
    transport(H,alpha=-1)
    transport(H,beta=-0.5)
    transport(H,gamma=-0.1)
    transport3(H)
    transport2(H)
