#Functions for TN propagation

import numpy as np
from scipy.linalg import expm
from numpy import linalg as linalg
import modules
#================================================
'''
kinpropgator: This function takes kinetic energy, timestep and hbar and gives kinetic propagator

expk = e^(-iKt/hbar)
'''
#================================================
def kinpropagator(kinetic, t, hbar):
    expk = {}
    for i in range(len(kinetic)):
        expk['%s'%i] = expm(-1j*kinetic['%s'%i]*t/hbar)
    return expk


#==================================================
'''
SVR: This function takes 1d array and threshold and gives number of elemnts which follows si/s0> threshold
'''
#==================================================
def SVR(se, thres):
    count=0
    for i in range(len(se)):
        if (se[i]/se[0])>thres:
            count = count + 1
    P = count
    return P


#====================================================
'''
Us: This function takes a matrix U, an 1d array and number of columns P, it will give a matrix of size (U.shape[0],P) where each column of input matrix is multiplied by corresponding array element. 
'''
#====================================================
def Us(Ue, se, P):
    U1 = Ue[:,0:P]*se[0:P]
    return U1
#====================================================


#====================================================
def U1W1(c, thres):
    Ue, se, We = linalg.svd(c, full_matrices= False)
    P = SVR(se, thres)
    U1 = Us(Ue, np.sqrt(se), P)
    # Fix Transponse for performance.
    W1T = Us(We.T, np.sqrt(se), P)
    W1 = W1T.T
    return U1, W1, se, P

#====================================================

def U1W1(c, thres=1E-7, P=0, flag=1):
    Ue, se, We = linalg.svd(c, full_matrices= False)
    if flag == 1:
        P = SVR(se, thres)
    else:
        P = P
    U1 = Us(Ue, np.sqrt(se), P)
    # Fix Transponse for performance.
    W1T = Us(We.T, np.sqrt(se), P)
    W1 = W1T.T
    return U1, W1, se, P

#====================================================
def AUWSP(c, thres=1E-7, Ndim=2, d=np.zeros(2, dtype=int), Q=np.zeros(1, dtype=int), flag=1):
    A = {}
    U = {}
    Ures = {}
    #Ures1 = {}
    W = {}
    s = {}
    P = np.zeros(Ndim-1, dtype=int)
    A['%s'%0] = c
    for i in range(Ndim-1):
        U['cx%s'%i], W['%s'%i], s['%s'%i], P[i] = U1W1(A['%s'%i], thres, Q[i], flag)
        A['%s'%(i+1)] = np.reshape(W['%s'%i], (P[i]*d[i+1], np.prod(d[(i+2):Ndim])), order = 'F')

    U['cx%s'%(Ndim-1)] = W['%s'%(Ndim-2)].T  
    for i in range(1, Ndim-1):
        Ures['cx%s'%i] = np.reshape(U['cx%s'%i], (P[i-1], d[i], P[i]), order = 'F')
    #Ures['cx%s'%i] = np.reshape(U['cx%s'%i], (d[i], P[i-1]*P[i]), order = 'F')
        #Ures1['cx%s'%i] = np.reshape(Ures['cx%s'%i], (P[i-1]*d[i], P[i]), order = 'F')
    
    Ures['cx%s'%0] = U['cx%s'%0]
    Ures['cx%s'%(Ndim-1)] = U['cx%s'%(Ndim-1)]
    #Ures1['cx%s'%0] = U['cx%s'%0]
    #Ures1['cx%s'%(Ndim-1)] = U['cx%s'%(Ndim-1)]
    return Ures, s, P
#====================================================


#====================================================

def MDQRSVD(vx, Ndim, d, thres):
    Q = {}
    R = {}
    Qres = {}
    #UR = {}
    sR = {}
    #WR = {}
    SL = {}
    SR = {}
    SRres = {}
    cx = {}
    vxres = {}
    Qp = np.zeros(Ndim-1, dtype=int)


    vxres['cx%s'%0] = vx['cx%s'%0]
    vxres['cx%s'%(Ndim-1)] = vx['cx%s'%(Ndim-1)]
        
    for j in range(1, Ndim-1):
        vxres['cx%s'%j] = np.reshape(vx['cx%s'%j],(-1,vx['cx%s'%j].shape[2]), order = 'F')
        
#QR of each cores
    for j in range(Ndim):
        print('vxres.shape = ',vxres['cx%s'%j].shape)
        Q['%s'%j], R['%s'%j] = linalg.qr(vxres['cx%s'%j])
        print('Q.shape, R.shape =', Q['%s'%j].shape, R['%s'%j].shape)

#Reshaping Qs
    for j in range(1, Ndim-1):
        print('d =', d)
        Qres['%s'%j] = np.reshape(Q['%s'%j], (R['%s'%(j-1)].shape[1] ,R['%s'%j].shape[0]*d[j]), order='F')
        print('Qres.shape =', Qres['%s'%j].shape)

    for j in range(1, Ndim-1):
        SL['%s'%j], SR['%s'%j], sR['%s'%j], Qp[j-1] = U1W1(R['%s'%(j-1)]@Qres['%s'%j], thres)
        print('Qp[j-1], thres =', Qp[j-1], thres)
        print('SL.shape, SR.shape =', SL['%s'%j].shape, SR['%s'%j].shape)
        SRres['%s'%j] = np.reshape(SR['%s'%j], (Qp[j-1]*d[j],Q['%s'%j].shape[1]), order='F')
        
     
    RS = R['%s'%(Ndim-2)]@R['%s'%(Ndim-1)].T
    SL['%s'%(Ndim-1)], SR['%s'%(Ndim-1)], sR['%s'%(Ndim-1)], Qp[Ndim-2] = U1W1(RS, thres)
    SRres['%s'%(Ndim-1)] = SR['%s'%(Ndim-1)]@Q['%s'%(Ndim-1)].T

    cx['cx%s'%0] = Q['%s'%0]@SL['%s'%1]
    for j in range(1, Ndim-1):
        cx['cx%s'%j] = np.reshape(SRres['%s'%j]@SL['%s'%(j+1)], (Qp[j-1], d[j], Qp[j]), order='F')
        #cx['cx%s'%j] = np.reshape(SRres['%s'%j]@SL['%s'%(j+1)], (d[j],Qp[j-1]*Qp[j]), order='F')
    cx['cx%s'%(Ndim-1)] = SRres['%s'%(Ndim-1)].T
    
    #vx['cx%s'%0] = cx['cx%s'%0]
    #print('vx0.shape =', vx['cx%s'%0].shape)
    #for j in range(1, Ndim-1):
     #   vx['cx%s'%j] = np.reshape(cx['cx%s'%j], (Qp[j-1]*d[j], Qp[j]), order = 'F')
      #  print('vx.shape =', vx['cx%s'%j].shape)
    print('Qp inside MQRSVD =', Qp)
    #vx['cx%s'%(Ndim-1)] = cx['cx%s'%(Ndim-1)]
    #print('vx(NDim-1).shape =', vx['cx%s'%(Ndim-1)].shape)

    #prodq, Mdq = recombine(vx, Ndim, d, Qp)
    
    #print(Qp)
    return cx, Qp, sR
#====================================================


#====================================================

def ini_wf(ini_x, fin_x, len_x, sig, mu):
    #mu = (ini_x + fin_x)/2
    y1 = np.linspace(ini_x, fin_x, len_x)
    y3 = (np.exp(-np.power(y1-mu,2.) / (2 * (sig**2.))))/(sig*np.sqrt(2*np.pi))
    X1 = np.array([[i] for i in y3])
    X2 = X1/linalg.norm(X1)
    return X2


#====================================================

def ini_gauss_wf(n, mu, sig):
    #mu = (ini_x + fin_x)/2
    y1 = np.linspace(0, 1, n)
    y3 = (np.exp(-np.power(y1-mu,2.) / (2 * (sig**2.))))/(sig*np.sqrt(2*np.pi))
    X1 = np.array([[i] for i in y3])
    X2 = X1/linalg.norm(X1)
    return X2

#====================================================

def md_gauss_fn(Ndim, d, mu, sig):
    X = {}
    X['cx%s'%0] = ini_gauss_wf(d[0], mu[0], sig[0])
    X['cx%s'%(Ndim-1)] = ini_gauss_wf(d[Ndim-1], mu[Ndim-1], sig[Ndim-1])
    for i in range(1,Ndim-1):
        y = ini_gauss_wf(d[i], mu[i], sig[i])
        X['cx%s'%i] = np.reshape(y, (1,len(y),1))
    return X
#===================================================
#====================================================
def Xd(ini_x, fin_x, d, sig, mu, Ndim):
    X = {}
    X['cx%s'%0] = ini_wf(ini_x[0], fin_x[0], d[0], sig[0], mu[0])
    X['cx%s'%(Ndim-1)] = ini_wf(ini_x[Ndim-1], fin_x[Ndim-1], d[Ndim-1], sig[Ndim-1], mu[Ndim-1])
    for i in range(1,Ndim-1):
        y = ini_wf(ini_x[i], fin_x[i], d[i], sig[i], mu[i])
        X['cx%s'%i] = np.reshape(y, (1,len(y),1))
    return X

#====================================================


#====================================================
def ini_Xd(X):
    Ndim = len(X)
    Xbig = X['cx%s'%0]
    for i in range(Ndim-1):
        Xbig1 = np.kron(Xbig, X['cx%s'%(i+1)])
        Xbig = Xbig1
    return Xbig
#====================================================


#====================================================

def recombine(vx, Ndim, d, Qp):
    prodres = vx['cx%s'%(Ndim-1)].T
    print('prodres ini.shape =', prodres.shape)
    for j in range (Ndim-2, 0, -1):
        print('rec j =', j)
        prodq = vx['cx%s'%j]@prodres
        print('prodq.shape, vx.shape =', prodq.shape, vx['cx%s'%j].shape)
        prodres = np.reshape(prodq, (Qp[j-1], d[j]*prodq.shape[1]), order= 'F')
        print('prodres.shape =', prodres.shape)
    prodq = vx['cx%s'%(0)]@prodres
    Mdq = np.reshape(prodq, (prodq.shape[0]*prodq.shape[1],1), order = 'F')
    return prodq, Mdq
#====================================================


#====================================================

def recombine3D(vx, Ndim):
    prodres = vx['cx%s'%(Ndim-1)].T
    print('prodres ini.shape =', prodres.shape)
    for j in range (Ndim-2, 0, -1):
        print('rec j =', j)
        prodq = np.einsum('ijk,kl->ijl' , vx['cx%s'%j], prodres, optimize='greedy')
        print('prodq.shape, vx.shape =', prodq.shape, vx['cx%s'%j].shape)
        #prodres = np.reshape(prodq, (Qp[j-1], d[j]*prodq.shape[2]), order= 'F')
        prodres = np.reshape(prodq, (prodq.shape[0], prodq.shape[1]*prodq.shape[2]), order= 'F')
        print('prodres.shape =', prodres.shape)
    prodq = vx['cx%s'%(0)]@prodres
    Mdq = np.reshape(prodq, (prodq.shape[0]*prodq.shape[1],1), order = 'F')
    return prodq, Mdq
#====================================================

def recombine_last_dim_faster(vx, Ndim):
    prodres = vx['cx%s'%(Ndim-1)].T
    print('prodres ini.shape =', prodres.shape)
    for j in range (Ndim-2, 0, -1):
        print('rec j =', j)
        prodq = np.einsum('ijk,kl->ijl' , vx['cx%s'%j], prodres, optimize='greedy')
        print('prodq.shape, vx.shape =', prodq.shape, vx['cx%s'%j].shape)
        #prodres = np.reshape(prodq, (Qp[j-1], d[j]*prodq.shape[2]), order= 'F')
        prodres = np.reshape(prodq, (prodq.shape[0], prodq.shape[1]*prodq.shape[2]))
        print('prodres.shape =', prodres.shape)
    prodq = vx['cx%s'%(0)]@prodres
    Mdq = np.reshape(prodq, (prodq.shape[0]*prodq.shape[1],1))
    return prodq, Mdq


#====================================================

def propd(U, M):
    Q = len(M[0])
    M1 = np.empty((len(U),0), dtype = complex)
    for k in range(Q):
            z1 = (U.T*M[:,k]).T
            M1 = np.append(M1, z1, axis = 1)
    return M1
#====================================================
def propnew(U, M):
    if len(U.shape)==2:
        M1 = np.reshape(np.einsum('ij,ik->ijk', U, M, optimize='greedy'), (U.shape[0],-1), 'F')
    elif len(U.shape)==3:
        M1 = np.reshape(np.einsum('ijk,ljm->iljkm', U, M, optimize='greedy'), (-1, U.shape[1], U.shape[2]*M.shape[2]), 'F')
    return M1
    

#====================================================
def prop3d(U, M):
    print('Inside prop3d')
    Q = U.shape[2]*M.shape[2]
    M3 = np.empty((0,U.shape[1],Q), dtype = complex)
    #print('M3.shape =', M3.shape)
    for l in range(M.shape[0]):
        for k in range(U.shape[0]):
            print('l, k inside prop3d =',l,k)
            M1 = propd(U[k,:,:], M[l,:,:])
            M2 = np.reshape(M1, (1,M1.shape[0], M1.shape[1]), 'F')
            #print('M2.shape =', M2.shape)
            M3 = np.append(M3, M2, axis=0)
    return M3
#====================================================

#====================================================

def potprop1(U, X, Ndim, d, thres, TNtype, regulate):
    fvec = {}
    cx = {}
    #fvecres = {}
    #vxres = {}
    #prodq = {}
    Qp = np.zeros(Ndim-1, dtype=int)
    #Qpx = np.zeros(Ndim-1, dtype=int)
    #Q = {}
    #R = {}
#Initialization 
    for i in range(Ndim):
        cx['cx%s'%i] = X['cx%s'%i]
     #   print('X[X%s%i].shape[1]', X['cx%s'%i].shape[1])
      #  print('X[X%s%i].shape, cx[0][cx%s%i].shape',X['cx%s'%i].shape, cx[0]['cx%s'%i].shape)

#propgating wf with pot prop
    for j in range(Ndim):
        fvec['cx%s'%j] = propnew(U['cx%s'%j], X['cx%s'%j])
    
    #fvec['cx%s'%0] = propd(U['cx%s'%0], X['cx%s'%0])
    #fvec['cx%s'%(Ndim-1)] = propd(U['cx%s'%(Ndim-1)], X['cx%s'%(Ndim-1)])
    #for j in range(1, Ndim-1):
        #fvec['cx%s'%j] = prop3d(U['cx%s'%j], X['cx%s'%j])
#Qpx

    #Qpx[0] = fvec['%s'%0].shape[1]
    #Qpx[Ndim-2] = fvec['%s'%(Ndim-1)].shape[1]
    #for j in range(1,Ndim-2):
        #Qpx[j] = fvec['%s'%j].shape[2]
        #print('Qpx =', Qpx)

#reshaping fvec to (a1x2, a2) form
#    fvecres['cx%s'%0] = fvec['%s'%0]
#    print('fvecres.shape =', fvecres['cx%s'%0].shape)
        
#    for j in range(1, Ndim-1):
        #print('Qpx =', Qpx)
        #fvecres['cx%s'%j] = np.reshape(fvec['%s'%j],(fvec['%s'%j].shape[0]*fvec['%s'%j].shape[1],fvec['%s'%j].shape[2]), order = 'F')
#        fvecres['cx%s'%j] = np.reshape(fvec['%s'%j],(-1,fvec['%s'%j].shape[2]), order = 'F')
#        print('fvecres[cx%s%j].shape =', fvecres['cx%s'%j].shape)
        
#    fvecres['cx%s'%(Ndim-1)] = fvec['%s'%(Ndim-1)]
#    print('fvecresNdim.shape =', fvecres['cx%s'%(Ndim-1)].shape)
        
    #prodfvec, _ = recombine(fvecres, Ndim, d, Qpx) 

#Direct prop without QRSVD(TNtype =1) or QRSVD after certain steps(TNtype = 3)
    if (TNtype == 1):
        for j in range(Ndim):
                #cx[j]['cx%s'%(i+1)] = fvec['%s'%j]
            cx['cx%s'%j] = fvec['cx%s'%j]
            #vxres['cx%s'%j] = fvecres['cx%s'%j]
         
        Qp[0] = fvec['cx%s'%0].shape[1]
        Qp[Ndim-2] = fvec['cx%s'%(Ndim-1)].shape[1]
        for j in range(1,Ndim-2):
            Qp[j] = fvec['cx%s'%j].shape[2]
        print('Qp =', Qp)

#prop with QRSVD(TNtype = 2)
    elif TNtype == 2:
        if regulate == 1:
            cx, Qp, _ = MDQRSVD(fvec, Ndim, d, thres)
        elif regulate == 2:
            cx, Qp = modified_round.round(fvec, Ndim, d, thres)
        
    #prod2, _ = recombine(vxres, Ndim, d, Qp) 
        
    #print('normcxvx =', np.linalg.norm(prodfvec-prod2))
    #print('normprod2 =', np.linalg.norm(prod2))
    return cx, Qp

#====================================================
def Loghamprop(Ures, X, Ndim, d, kinetic, T, t):
    Unorm = {}
    Uphase = {}
    PhaU = {}
    normU = {}
    steps = int(T/t) 
    tp = np.linspace(0,T,steps)
    HamU = [dict() for i in range(Ndim)]
    cx = [dict() for i in range(len(tp))]
    for i in range(Ndim):
        Unorm['cx%s'%i] = np.abs(Ures['cx%s'%i])
        Uphase['cx%s'%i] = np.angle(Ures['cx%s'%i])
        PhaU['cx%s'%i] = modules.funaddall(Uphase['cx%s'%i],Uphase['cx%s'%i]) 
        normU['cx%s'%i] = modules.funprodall(Unorm['cx%s'%i],Unorm['cx%s'%i]) 
        
    for i in range(len(tp)):
        for j in range(Ndim):
            cx[i]['cx%s'%j] = np.zeros(PhaU['cx%s'%j].shape, dtype=complex)

    for i in (0,Ndim-1):
        for j in range(PhaU['cx%s'%i].shape[1]):
            HamU[i][j] = kinetic['%s'%i] - np.diag(PhaU['cx%s'%i][:,j])/t
            BU = modules.prop_eigbasis(X['cx%s'%i], HamU[i][j], tp)
            for l in range(len(tp)):
                cx[l]['cx%s'%i][:,j] = BU[:,l] 
    
    for i in range(1,Ndim-1):
        for j in range(PhaU['cx%s'%i].shape[0]):
            for k in range(PhaU['cx%s'%i].shape[2]):
                HamU[i].setdefault(j, {})[k] = kinetic['%s'%i] - np.diag(PhaU['cx%s'%i][j,:,k])/t
                BU = modules.prop_eigbasis(X['cx%s'%i], HamU[i][j][k], tp)
                for l in range(len(tp)):
                    cx[l]['cx%s'%i][j,:,k] = BU[:,l] 

    return cx
#====================================================
        
def first_order_Hamiltonian(Ures, Ndim, kinetic, t):
    Unorm = {}
    Uphase = {}
    PhaU = {}
    normU = {}
    #steps = int(T/t) 
    #tp = np.linspace(0,T,steps)
    HamU = [dict() for i in range(Ndim)]
    #cx = [dict() for i in range(len(tp))]
    for i in range(Ndim):
        Unorm['cx%s'%i] = np.abs(Ures['cx%s'%i])
        Uphase['cx%s'%i] = np.angle(Ures['cx%s'%i])
        #if first_order == 1
        PhaU['cx%s'%i] = Uphase['cx%s'%i]
        normU['cx%s'%i] = Unorm['cx%s'%i] 
        #PhaU['cx%s'%i] = modules.funaddall(Uphase['cx%s'%i],Uphase['cx%s'%i]) 
        #normU['cx%s'%i] = modules.funprodall(Unorm['cx%s'%i],Unorm['cx%s'%i]) 
        
    #for i in range(len(tp)):
        #for j in range(Ndim):
            #cx[i]['cx%s'%j] = np.zeros(PhaU['cx%s'%j].shape, dtype=complex)

    for i in (0,Ndim-1):
        for j in range(PhaU['cx%s'%i].shape[1]):
            HamU[i][j] = kinetic['%s'%i] - np.diag(PhaU['cx%s'%i][:,j])/t
            #BU = modules.prop_eigbasis(X['cx%s'%i], HamU[i][j], tp)
            #BUnorm = np.c_[normU['cx%s'%i][:,j]]*BU
            #for l in range(len(tp)):
                #cx[l]['cx%s'%i][:,j] = BUnorm[:,l] 
    
    for i in range(1,Ndim-1):
        for j in range(PhaU['cx%s'%i].shape[0]):
            for k in range(PhaU['cx%s'%i].shape[2]):
                HamU[i].setdefault(j, {})[k] = kinetic['%s'%i] - np.diag(PhaU['cx%s'%i][j,:,k])/t
                #BU = modules.prop_eigbasis(X['cx%s'%i], HamU[i][j][k], tp)
                #BUnorm = np.c_[normU['cx%s'%i][j,:,k]]*BU
                #for l in range(len(tp)):
                    #cx[l]['cx%s'%i][j,:,k] = BUnorm[:,l] 

    return PhaU, normU, HamU
#==================================================== 
def prop_logham(normU, HamU, Ndim, X, t):
    tp = np.linspace(t,t,1)
    print(tp)
    cx = {}
    for i in range(Ndim):
        cx['cx%s'%i] = np.zeros(normU['cx%s'%i].shape, dtype=complex)
    
    for i in (0,Ndim-1):
        for j in range(normU['cx%s'%i].shape[1]):
            BU = modules.prop_eigbasis(X['cx%s'%i], HamU[i][j], tp)
            cx['cx%s'%i][:,j] = (np.c_[normU['cx%s'%i][:,j]]*BU)[:,0]
    
    for i in range(1,Ndim-1):
        for j in range(normU['cx%s'%i].shape[0]):
            for k in range(normU['cx%s'%i].shape[2]):
                BU = modules.prop_eigbasis(X['cx%s'%i], HamU[i][j][k], tp)
                cx['cx%s'%i][j,:,k] = (np.c_[normU['cx%s'%i][j,:,k]]*BU)[:,0]
     
    return cx
    
#====================================================

def TNMDprop(Ures, X, Ndim, d, P, expk, thres, steps, TNtype, regulate, DEBUG, DEBUG1):
    x = [dict() for i in range(steps+1)]
    cx = [dict() for i in range(steps+1)]
    #vxres = {}
    vkx = {}
    Qp = {}
    #Qpx = {}
    for i in range(steps+1):
        Qp['Qp%s'%i] = np.zeros(Ndim-1, dtype=int)
     #   Qpx['Qp%s'%i] = np.zeros(Ndim-1, dtype=int)
    
#Initialization 
    for i in range(Ndim):
        x[0]['cx%s'%i] = X['cx%s'%i]
        cx[0]['cx%s'%i] = X['cx%s'%i]
        if DEBUG:
            print('X[X%s%i].shape, cx[0][cx%s%i].shape',X['cx%s'%i].shape, cx[0]['cx%s'%i].shape)
     
    Qp['Qp%s'%0][0] = X['cx%s'%0].shape[1]
    Qp['Qp%s'%0][Ndim-2] = X['cx%s'%(Ndim-1)].shape[1]
    for i in range(1,Ndim-2):
        Qp['Qp%s'%0][i] = X['cx%s'%i].shape[2]
    
    #Qpx['Qp%s'%0][0] = X['cx%s'%0].shape[1]
    #Qpx['Qp%s'%0][Ndim-2] = X['cx%s'%(Ndim-1)].shape[1]
    #for i in range(1,Ndim-2):
     #   Qpx['Qp%s'%0][i] = X['cx%s'%i].shape[2]
    
#Propagation
    for i in range(steps):
        if DEBUG1:
            print('steps inside TNMDprop=', i)
        #x[i+1], vxres, Qpx['Qp%s'%(i+1)] = potprop1(Ures, cx[i], Ndim, d, thres, TNtype)
        x[i+1], _ = potprop1(Ures, cx[i], Ndim, d, thres, TNtype, regulate)
        
        vkx['cx%s'%0] = expk['%s'%0]@x[i+1]['cx%s'%0] 
        vkx['cx%s'%(Ndim-1)] = expk['%s'%(Ndim-1)]@x[i+1]['cx%s'%(Ndim-1)] 
        if DEBUG1:
            print('vkx0.shape =', vkx['cx%s'%0].shape)
            print('vkx(Ndim-1).shape =', vkx['cx%s'%(Ndim-1)].shape)
        for j in range(1,Ndim-1):
            vkx['cx%s'%j] = np.einsum('ij,kjl->kil', expk['%s'%j], x[i+1]['cx%s'%j])
            if DEBUG1:
                print('vkx%s.shape ='%j, vkx['cx%s'%j].shape)
        cx[i+1], Qp['Qp%s'%(i+1)] = potprop1(Ures, vkx, Ndim, d, thres, TNtype, regulate)
    return cx, Qp


#====================================================



#====================================================

def temptrot1(c, X1, kinetic, steps):
    trot = {}
    trot['cx%s'%0] = X1
    CVX2 = X1
    for i in range(steps):
        CVX = c*CVX2
        print('CVX.shape =', CVX.shape)
        k1 = np.einsum('ij,jkl->ikl', kinetic['%s'%0], CVX)
        print('k1.shape =', k1.shape)
        k2 = np.einsum('ij,kjl->kil', kinetic['%s'%1], k1)
        print('k2.shape =', k2.shape)
        k3 = np.einsum('ij,klj->kli', kinetic['%s'%2], k2)
        print('k3.shape =', k3.shape)
        CVX2 = c*k3
        trot['cx%s'%(i+1)] = CVX2
        print('normCVX =', np.linalg.norm(CVX2))
    return trot


def trotprop(c, expk, X1, Ndim, d, steps):
    trot = {}
    trot['cx%s'%0] = X1
    CVX2 = X1
    for i in range(steps):
        CVX = c*CVX2
        for j in range(Ndim):
            k = expk['%s'%j]@CVX
            if j < Ndim-1:
                print(j)
                CVX = np.reshape(k.T, (d[j+1],-1), 'F')
            elif j == Ndim-1:
                print(j, 'inside')
                CVX = np.reshape(k.T, (d[0],-1), 'F')
                print(d[0], CVX.shape)
        CVX2 = c*CVX
        trot['cx%s'%(i+1)] = CVX2
        print('normCVX =', np.linalg.norm(CVX2))
    return trot, CVX2

def trotpropnew(c, expk, X0, Ndim, d, T, t):
    steps = int(T/t) 
    X1 = np.reshape(X0, (d[0], np.prod(d[1:])))
    trot = {}
    trotMq = np.zeros((np.prod(d), 0), dtype=complex)
    trot['cx%s'%0] = X1
    CVX2 = X1
    for i in range(steps):
        CVX = c*CVX2
        for j in range(Ndim):
            k = expk['%s'%j]@CVX
            if j < Ndim-1:
                print(j)
                CVX = np.reshape(k.T, (d[j+1],-1), 'F')
            elif j == Ndim-1:
                print(j, 'inside')
                CVX = np.reshape(k.T, (d[0],-1), 'F')
                print(d[0], CVX.shape)
        CVX2 = c*CVX
        trot['cx%s'%(i+1)] = CVX2
        Mtrot = np.reshape(CVX2, (np.prod(d),1))
        trotMq = np.hstack((trotMq, Mtrot))
        print('normCVX =', np.linalg.norm(CVX2))
    return trot, CVX2, trotMq

#===========================================================================
def direct_diagonalization(kinetic,V,Ndim,X0,T,t):    
    if (Ndim==2):
        H = np.kron(kinetic['%s'%0], np.eye(len(kinetic['%s'%1])))+np.kron(np.eye(len(kinetic['%s'%0])),kinetic['%s'%1])+np.diagflat(V)

    if (Ndim==3):
        H = np.kron(np.kron(kinetic['%s'%0], np.eye(kinetic['%s'%1].shape[0])), np.eye(kinetic['%s'%2].shape[0]))+np.kron(np.kron(np.eye(kinetic['%s'%0].shape[0]),kinetic['%s'%1]), np.eye(kinetic['%s'%2].shape[0]))+np.kron(np.kron(np.eye(kinetic['%s'%0].shape[0]),np.eye(kinetic['%s'%1].shape[0])), kinetic['%s'%2])+np.diagflat(V)

    steps = int(T/t) 
    tp = np.linspace(0,steps*t,steps+1)
    #EH,VH = linalg.eigh(H)
    Mp = modules.prop_eigbasis(X0,H,tp)
    
    return H, Mp

