"""Scientific Computation Project 3, part 2
Your CID here
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import sparse,signal,spatial

def microbes(phi,kappa,mu,L = 1024,Nx=1024,Nt=1201,T=600,display=False):
    """
    Question 2.2
    Simulate microbe competition model

    Input:
    phi,kappa,mu: model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of f when true

    Output:
    f,g: Nt x Nx arrays containing solution
    """

    #generate grid
    L = 1024
    x = np.linspace(0,L,Nx)
    dx = x[1]-x[0]
    dx2inv = 1/dx**2

    def RHS(y,t,k,r,phi,dx2inv):
        #RHS of model equations used by odeint

        n = y.size//2

        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        #Construct RHS
        R = f/(f+phi)
        dfdt = d2f + f[1:-1]*(1-f[1:-1])- R[1:-1]*g[1:-1]
        dgdt = d2g - r*k*g[1:-1] + k*R[1:-1]*g[1:-1]
        dy = np.zeros(2*n)
        dy[1:n-1] = dfdt
        dy[n+1:-1] = dgdt

        #Enforce boundary conditions
        a1,a2 = -4/3,-1/3
        dy[0] = a1*dy[1]+a2*dy[2]
        dy[n-1] = a1*dy[n-2]+a2*dy[n-3]
        dy[n] = a1*dy[n+1]+a2*dy[n+2]
        dy[-1] = a1*dy[-2]+a2*dy[-3]

        return dy


    #Steady states
    rho = mu/kappa
    F = rho*phi/(1-rho)
    G = (1-F)*(F+phi)
    y0 = np.zeros(2*Nx) #initialize signal
    y0[:Nx] = F
    y0[Nx:] = G + 0.01*np.cos(10*np.pi/L*x) + 0.01*np.cos(20*np.pi/L*x)

    t = np.linspace(0,T,Nt)

    #compute solution
    #print("running simulation...")
    y = odeint(RHS,y0,t,args=(kappa,rho,phi,dx2inv),rtol=1e-6,atol=1e-6)
    f = y[:,:Nx]
    g = y[:,Nx:]
   # print("finished simulation")
    if display:
        plt.figure()
        plt.contour(x,t,f)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of f')


    return f,g


def newdiff(f,h):
    """
    Question 2.1 i)
    Input:
        f: array whose 2nd derivative will be computed
        h: grid spacing
    Output:
        d2f: second derivative of f computed with compact fd scheme
    """

    N=len(f)
    #Coefficients for compact fd scheme
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140
    a_1=c/(9*h**2)
    a_2=b/(4*h**2)
    a_3=a/(h**2)
    s=-2*(a_1+a_2+a_3)
    #construct LHS matrix
    v=np.ones(N)
    v_1=np.append(alpha*v[2:],[10])
    v_2=np.append([10],alpha*v[2:])
    A=sparse.diags([v_1,v,v_2],[-1,0,1]).toarray()
    #construct RHS matrices
    v_0=np.append(np.append([145/(12*h**2)],s*v[2:]),[145/(12*h**2)])

    v_minus_1=np.append(a_3*v[2:],[-76/(3*h**2)])
 
    v_minus_2=np.append(a_2*v[3:],[29/(2*h**2)])
  
    v_minus_3=np.append(a_1*v[4:],[-4/(3*h**2)])
  
    v_minus_4=np.append(0*v[5:],[1/(12*h**2)])
  
    v_11=np.append([-76/(3*h**2)],a_3*v[2:])
  
    v_22=np.append([29/(2*h**2)],a_2*v[3:])
   
    v_3=np.append([-4/(3*h**2)],a_1*v[4:])
   
    v_4=np.append([1/(12*h**2)],0*v[5:])
    v_n_minus2=np.array([0,a_2])
    v_n_minus3=np.array([0,a_1,a_1])
    v_minus_n_minus3=np.array([a_1,a_1,0])
    v_minus_n_minus2=np.array([a_2,0])
    B=sparse.diags([v_minus_n_minus3,v_minus_n_minus2,v_minus_4,v_minus_3,v_minus_2,v_minus_1,v_0,v_11,v_22,v_3,v_4,v_n_minus2,v_n_minus3],[-N+3,-N+2,-4,-3,-2,-1,0,1,2,3,4,N-2,N-3])
    H=np.matmul(B.toarray(),f)

    #solve equations
    d2f=sparse.linalg.spsolve(A,H)
  
     
    return d2f

def analyzefd(inputs=()):
    """
    Question 2.1 ii)
    Add input/output as needed

    """
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140
    kh=np.linspace(0,np.pi,100)
    l=np.linspace(0,2*np.pi,10000)
    f=np.sin(l)
    f2=newdiff(f,np.pi/5000)
    #wavenumber analysos
    fig1=plt.figure()
    plt.plot(kh,-(kh)**2,color='green',label='Exact')
    plt.plot(kh,2*np.cos(kh)-2,color='red',label='2nd Ordered Centred FD')
    plt.plot(kh,(2*c/9*(-1+np.cos(3*kh))+b/2*(-1+np.cos(2*kh))\
              +2*a*(-1+np.cos(kh)))/(1+2*alpha*np.cos(kh)),color='blue',linestyle='dashed',label='Compact FD')
    plt.xlabel('kh',fontsize=20)
    plt.ylabel('Modified Wave Number',fontsize=16)
    plt.title('Wave Number Analysis',fontsize=16)
    plt.legend()
    fig1.show()
    
    #checks accuracy of compact with respect to sine function
    fig2=plt.figure()
    plt.plot(f2[8:-8],label='2nd derivative of sine function, FD Scheme',color='red')
    plt.plot(-np.sin(l),color='blue',linestyle='dashed',label='Exact')
    plt.title('Compact FD Scheme for Sine Function',fontsize=35)
    plt.legend(loc='lower right',fontsize=25)
    fig2.show()
    
    #check error of the scheme
    fig3=plt.figure()
    plt.semilogy(l,np.abs(-np.sin(l)-f2))
    plt.xlabel('x',fontsize=20)
    plt.ylabel('Error',fontsize=20)
    plt.title('Error with respect to each gridpoint',fontsize=16)
    fig3.show()
    
    print('Mean error is %s' %np.mean(np.abs(-np.sin(l)-f2)))
    
def dynamics(data,r):
    """
    Question 2.2
    Add input/output as needed

    """
    #fractal dimension for f,g dynamic system
    for k in [1.5,1.7,2]:
        f,g=microbes(phi=0.3,kappa=k,mu=0.4*k)
        
        f=f[700:,100]
        g=g[700:,100]
        A=np.vstack([f,g])
        n=len(f)
        D=spatial.distance.pdist(A.T)
        eps=np.linspace(0,0.13,200)
        C=np.zeros_like(eps)
        for i in range(len(eps)):
              B=D[D<eps[i]]
              C[i]=(2/(n*(n-1)))*B.size
        slope,intercept=np.polyfit(np.log(eps[3:20]),np.log(C[3:20]),1) 
        plt.figure()
        plt.plot(eps,C,'x--')
        plt.plot(eps[3:20],np.exp(np.log(eps[3:20])*slope+intercept),'x--',color='red',label='Fractal dimension is %s'%slope)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\epsilon$',fontsize=20)
        plt.ylabel(r'C($\epsilon$)',fontsize=20)
        plt.title('Correlation sum for k=%s'%k,fontsize=20)
        plt.legend(loc='lower right',fontsize=30)
    plt.show()

    #fractal dimension for 5pi/4
    plt.figure()
    for ri in [0,2,5,10,15]:
        w,P=signal.welch(data[ri,180,:])
        f = w[P==P.max()]
        tau=int(1//(5*f)[0])
        data_delayed=time_embedding(data[ri,180,:],tau,4)
        D=spatial.distance.pdist(data_delayed)
        eps=np.linspace(0.1,100,200)
        C=np.zeros_like(eps)
        
        for i in range(len(eps)):
              B=D[D<eps[i]]
              C[i]=(2/(n*(n-1)))*B.size
        slope,intercept=np.polyfit(np.log(eps[3:20]),np.log(C[3:20]),1) 
       
        plt.plot(eps,C,'x--',label=r'Fractal dimension %s and r = %s'%(slope,r[ri]))
      
        plt.xlabel(r'$\epsilon$',fontsize=20)
        plt.ylabel(r'C($\epsilon$)',fontsize=20)
        plt.xscale('log')
        plt.yscale('log') 
        

    plt.title(r'Correlation sum for $\theta=\frac{5\pi}{4}$ ',fontsize=20)
    plt.legend(fontsize=16,loc='lower right')
    plt.show()
  
    #fractal dimension for 3pi/4
    plt.figure()
    for ri in [0,2,5,10,15]:
        w,P=signal.welch(data[ri,108,:])
        f = w[P==P.max()]
        tau=int(1//(5*f)[0])
        data_delayed=time_embedding(data[ri,108,:],tau,5)
        D=spatial.distance.pdist(data_delayed)
        eps=np.linspace(0.01,100,100)
        C=np.zeros_like(eps)
        
        for i in range(len(eps)):
              B=D[D<eps[i]]
              C[i]=(2/(n*(n-1)))*B.size
        slope,intercept=np.polyfit(np.log(eps[3:20]),np.log(C[3:20]),1) 
       
        plt.plot(eps,C,'x--',label=r'Fractal dimension %s and r = %s'%(slope,r[ri]))
        plt.xlabel(r'$\epsilon$',fontsize=20)
        plt.ylabel(r'C($\epsilon$)',fontsize=20)
        plt.xscale('log')
        plt.yscale('log') 
        
    plt.title(r'Correlation sum for $\theta=\frac{3\pi}{4}$ ',fontsize=20)
    plt.legend(fontsize=16,loc='upper left')
    plt.show()
    
    #fractal dimension for pi/4
    plt.figure()
    for ri in [0,2,5,10,15]:
        w,P=signal.welch(data[ri,36,:])
        f = w[P==P.max()]
        tau=int(1//(5*f)[0])
        data_delayed=time_embedding(data[ri,36,:],tau,2)
        D=spatial.distance.pdist(data_delayed)
        eps=np.linspace(0.1,10,100)
        C=np.zeros_like(eps)
        
        for i in range(len(eps)):
              B=D[D<eps[i]]
              C[i]=(2/(n*(n-1)))*B.size
        slope,intercept=np.polyfit(np.log(eps[3:20]),np.log(C[3:20]),1) 
       
        plt.plot(eps,C,'x--',label=r'Fractal dimension %s and r = %s'%(slope,r[ri]))

        plt.xlabel(r'$\epsilon$',fontsize=20)
        plt.ylabel(r'C($\epsilon$)',fontsize=20)
        plt.xscale('log')
        plt.yscale('log') 
       
    plt.title(r'Correlation sum for $\theta=\frac{\pi}{4}$ ',fontsize=20)
    plt.legend(fontsize=20,loc='lower right')
    plt.show()
    
    
    for k in [1.5,1.7,2]:
        f,g=microbes(phi=0.3,kappa=k,mu=0.4*k)
        plt.figure()
        plt.plot(f[1::2,100],f[2::2,100],'.')
        plt.xlabel(r'$x_{n}$',fontsize=25)
        plt.ylabel(r'$x_{n+1}$',fontsize=25)
        plt.title(r'k=%s for f'%k,fontsize=30)
    plt.show()
    
    
    for k in [1.5,1.7,2]:
        f,g=microbes(phi=0.3,kappa=k,mu=0.4*k)
        plt.figure()
        plt.plot(g[1::2,100],g[2::2,100],'.')
        plt.xlabel(r'$x_{n}$',fontsize=25)
        plt.ylabel(r'$x_{n+1}$',fontsize=25)
        plt.title(r'k=%s for g'%k,fontsize=30)
    plt.show()
   
    #phase space plot
    for k in [1.5,1.6,1.7,1.8,2]:
        f,g=microbes(phi=0.3,kappa=k,mu=0.4*k,T=2000,Nt=2402)
        plt.figure()
        plt.plot(f[:,100],g[:,100])
        plt.title(r'k=%s'%k,fontsize=40)
    plt.show()

aszz       
def time_embedding(timeseries,tau,m):
    embedding=np.zeros((timeseries.shape[0],m))
    embedding[:,0]=timeseries
    for i in range(1,m):
        rolled_t=np.roll(timeseries,i*tau)
        embedding[:,i]=rolled_t
    return embedding
        
        




    
    
if __name__=='__main__' :
     x=None
