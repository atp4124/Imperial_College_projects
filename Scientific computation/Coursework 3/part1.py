"""Scientific Computation Project 3, part 1
Your CID here
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def hfield(r,th,h,levels=50):
    """Displays height field stored in 2D array, h,
    using polar grid data stored in 1D arrays r and th.
    Modify as needed.
    """
    thg,rg = np.meshgrid(th,r)
    xg = rg*np.cos(thg)
    yg = rg*np.sin(thg)
    plt.figure()
    plt.contourf(xg,yg,h,levels)
    plt.axis('equal')
    return None

def repair1(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data

    S = set()
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
            for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    Bfac = 0.0
                    Asum = 0

                    for j in mlist[m]:
                        Bfac += B[n,j]**2
                        Rsum = 0
                        for k in range(p):
                            if k != n: Rsum += A[m,k]*B[k,j]
                        Asum += (R[m,j] - Rsum)*B[n,j]

                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m<p:
                   #update B[m,n]
                    Afac=0.0
                    Bsum= 0 
                    for i in nlist[n]:
                        Afac+=A[i,m]**2
                        Rsum=0
                        for k in range(p):
                            if k!=m: Rsum+=A[i,k]*B[k,n]
                        Bsum+=(R[i,n]-Rsum)*A[i,m]
                    B[m,n]=Asum/(Afac+l) #new B[m,n]
        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if z%10==0: print("z,dA,dB=",z,dA[z],dB[z])


    return A,B


def repair2(R,p,l=1.0,niter=10,inputs=()):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R. Efficient and complete version of repair1.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
 #problem setup
    R0 = R.copy()
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data

   

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j)
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a):
    
             for n in np.random.permutation(b):
                if n < p: #Update A[m,n]
                    
                   Bfac=np.dot(B[n,mlist[m]],B[n,mlist[m]])
                   A_n=np.concatenate((A[:,:n],A[:,n+1:]),axis=1)
                   B_n=np.concatenate((B[:n,mlist[m]],B[n+1:,mlist[m]]),axis=0)
                   Rsum_1=R[m,mlist[m]]-np.matmul(A_n[m,:],B_n)
                   Asum=np.dot(Rsum_1,B[n,mlist[m]])
                   A[m,n]=Asum/(Bfac+l)                  
                if m<p:
                    #Add code here to update B[m,n]
                    Afac=np.dot(A[nlist[n],m],A[nlist[n],m])
                    A_m=np.concatenate((A[nlist[n],:m],A[nlist[n],m+1:]),axis=1)
                    B_m=np.concatenate((B[:m,:],B[m+1:,:]),axis=0)
                    Rsum=R[nlist[n],n]-np.matmul(A_m,B_m[:,n])
                    Bsum=np.dot(Rsum,A[nlist[n],m])
                    B[m,n]=Bsum/(Afac+l)
        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if z%10==0: print("z,dA,dB=",z,dA[z],dB[z])

    return A,B


def outwave(r0):
    """
    Question 1.2i)
    Calculate outgoing wave solution at r=r0
    See code/comments below for futher details
        Input: r0, location at which to compute solution
        Output: B, wave equation solution at r=r0

    """
    A = np.load('data2.npy')
    r = np.load('r.npy')
    th = np.load('theta.npy')

    Nr,Ntheta,Nt = A.shape
    B = np.zeros((Ntheta,Nt))

    return None

def analyze1(timeseries,r,theta):
    """
    Question 1.2ii)
    Add input/output as needed

    """
    N=119
    (fig1,ax1)=plt.subplots()
    markers=iter(["x","o","x"])
    lines=iter(["--","--",""])
    #plot of the frequencies for different values of theta
    for i in [36,108,180]:
        
        c=np.fft.fft(timeseries[0,i,:])
        c=np.fft.fftshift(c)/N
        n=np.arange(-N/2,N/2)
        ax1.plot(n,np.abs(c),'x',marker=next(markers),linestyle=next(lines))
        ax1.set_xlabel('mode number, n',fontsize=20)
        ax1.set_ylabel('$|c_n|$',fontsize=20)
    ax1.legend((r'$\theta=\frac{\pi}{4}$',r'$\theta=\frac{3\pi}{4}$',r'$\theta=\frac{5\pi}{4}$'),fontsize=16)
    
   #mean value of the amplitudes of the frequencies for different values of r
    vec=[]
    vec2=[]
    vec3=[]
    (fig3,ax3)=plt.subplots()
 
    for i in range(300):
        c1=np.fft.fft(timeseries[i,36,:])
        c1=np.fft.fftshift(c1)/N
        cc1=np.abs(c1)
        mean_amp1=np.mean(cc1[cc1>0.001])
        vec.append(mean_amp1)
        c2=np.fft.fft(timeseries[i,108,:])
        c2=np.fft.fftshift(c2)/N
        cc2=np.abs(c2)
        mean_amp2=np.mean(cc2[cc2>0.001])
        vec2.append(mean_amp2)
        c3=np.fft.fft(timeseries[i,180,:])
        c3=np.fft.fftshift(c3)/N
        cc3=np.abs(c3)
        mean_amp3=np.mean(cc3[cc3>0.001])
        vec3.append(mean_amp3)
    ax3.plot(r,vec,'x')
    ax3.plot(r,vec2,'x')
    ax3.plot(r,vec3,'x')
    ax3.set_xlabel('r',fontsize=20)
    ax3.set_ylabel('Mean value of amplitudes above 0',fontsize=15 )
    ax3.legend((r'$\theta=\frac{\pi}{4}$',r'$\theta=\frac{3\pi}{4}$',r'$\theta=\frac{5\pi}{4}$'),fontsize=16)
    

    #plot the mean of the timeseries for different values of r
    m1=[]
    m2=[]
    m3=[]
    (fig4,ax4)=plt.subplots()
    for i in range(300):
        m1.append(np.mean(timeseries[i,36,:]))
        m2.append(np.mean(timeseries[i,108,:]))
        m3.append(np.mean(timeseries[i,180,:]))
    ax4.plot(r,m1)
    ax4.plot(r,m2)
    ax4.plot(r,m3)
    ax4.set_xlabel('r',fontsize=20)
    ax4.set_ylabel('Mean value of timeseries',fontsize=15)
    ax4.legend((r'$\theta=\frac{\pi}{4}$',r'$\theta=\frac{3\pi}{4}$',r'$\theta=\frac{5\pi}{4}$'),fontsize=16)
    
    #plot the variance of the timeseries for different values of r    
    m4=[]
    m5=[]
    m6=[]
    (fig5,ax5)=plt.subplots()
    for i in range(300):
        m4.append(np.var(timeseries[i,36,:]))
        m5.append(np.var(timeseries[i,108,:]))
        m6.append(np.var(timeseries[i,180,:]))
    ax5.plot(r,m4)
    ax5.plot(r,m5)
    ax5.plot(r,m6)
    ax5.set_xlabel('r',fontsize=20)
    ax5.set_ylabel('Variance value of timeseries',fontsize=15)
    ax5.legend((r'$\theta=\frac{\pi}{4}$',r'$\theta=\frac{3\pi}{4}$',r'$\theta=\frac{5\pi}{4}$'),fontsize=16)
    fig1.show()
    fig3.show()
    fig4.show()
    fig5.show()

    for i in [36,108,180]:
        plt.figure()
        for j in range(20):
            plt.plot(timeseries[j,i,:])
            plt.ylabel(r'Height',fontsize=25)
            plt.xlabel(r'Time point',fontsize=25)
            plt.title(r'Theta is %s'%theta[i])
    plt.show()

        
def reduce(H,per, inputs=()):
    """
    Question 1.3: Construct one or more arrays from H
    that can be used by reconstruct
    Input:
        H: 3-D data array
        per:percentage
    Output:
        arrays: a tuple containing the arrays produced from H
    """

    #list containing arrays produced from H
    arrays=[0,0,0]
    m,n,t = H.shape
    #list to store the results from SVD for each slice
    U_list=[]
    G_list=[]
    means_list=[]
    #iterate through the time dimension
    for i in range(H.shape[2]):
        #slice H through each time point
         X=H[:,:,i]
         Means=np.outer(np.ones((m,1)),X.mean(axis=0))
         new_data2=X-Means
         U,S,VT=np.linalg.svd(new_data2)
         G=np.dot(U.T,new_data2)
        #in order to find the number of components we need, we use cumulative sum to see where it is less
        #than what we need
         var_exp=[(i/sum(S))*100 for i in sorted(S,reverse=True)]
         ind=np.argwhere(np.cumsum(var_exp)<per)
         ind=ind[len(ind)-1][0]
         #append the components for each slice to our lists
         U_list.append(U[:,0:ind])
         G_list.append(G[0:ind,:])
         means_list.append(Means)
    
    print(r'Dimension of U is %s by %s' %(len(U_list),U_list[0].shape))
    print(r'Dimension of G is %s by %s' % (len(G_list),G_list[0].shape))
    print(r'Percentage we wish to preserve  is %s'%per)
    arrays[0]=U_list
    arrays[1]=G_list
    arrays[2]=means_list 
    #return arrays
    

def reconstruct(arrays,inputs=()):
    """
    Question 1.3: Generate matrix with same shape as H (see reduce above)
    that has some meaningful correspondence to H
    Input:
        arrays: tuple generated by reduce
        inputs: can be used to provide other input as needed
    Output:
        Hnew: a numpy array with the same shape as H
    """
    Hnew=np.zeros((300,289,119))
    #iterate through each slice
    for i in range(119):
        #get the results from the reduce function
        U,G,Means=arrays[0][i],arrays[1][i],arrays[2][i]
        Hnew_prime=np.zeros((len(U),len(G[0])))
        for j in range(len(G)):
            #recuperate our data
            Hnew_prime=Hnew_prime+np.outer(U[:,j],G[j,:])
        Hnew[:,:,i]=Hnew_prime+Means
    return Hnew


if __name__=='__main__':
    x=None
