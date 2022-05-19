import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import inv
from scipy.ndimage import laplace
from scipy.signal import convolve


def gauss2d(X, Y, mu, sigma):
    
    R = np.sqrt(X**2 + Y**2)
    g = np.exp( -(R - mu)**2/(2*sigma**2) )
    
    return g / np.sum(g)


def disk2d(X, Y, T):
        
    R = np.sqrt(X**2 + Y**2)
    d = np.where(R<T,1,0)
    
    return d / np.sum(d)


def convolution_matrix(K, I):
    #Part 1: generate doubly blocked toeplitz matrix
    
    # calculate sizes
    K_row_num, K_col_num = K.shape
    I_row_num, I_col_num = I.shape
    R_row_num = K_row_num + I_row_num - 1
    R_col_num = K_col_num + I_col_num - 1
    # pad the kernel
    K_pad = np.pad(K, ((0,R_row_num - K_row_num),
                      (0,R_col_num - K_col_num)), 
                  'constant', constant_values= 0)
    # Assemble the list of Toeplitz matrices
    toeplitz_list = []
    for i in range(R_row_num):
        c = K_pad[i,:]
        r = np.r_[c[0],np.zeros(I_col_num-1)]
        toeplitz_list.append(toeplitz(c,r).copy())
    # make a matrix with the indices of the block
    # of the doubly blocked Toeplitz matrix
    c = np.array(range(R_row_num))
    r = np.r_[c[0], c[-1:1:-1]]
    doubly_indices = np.array(toeplitz(c,r).copy())
    # assemble the doubly blocked toeplitz matrix
    toeplitz_m = []
    for i in range(R_row_num):
        row = []
        for j in range(I_row_num):
            row.append(toeplitz_list[doubly_indices[i,j]])
        row = np.hstack(row)
        toeplitz_m.append(row)
    toeplitz_m = np.vstack(toeplitz_m)
    
    #Part 2: pad and flatten input image 
    
    dim = np.empty(2).astype(int)

    dim[0] = (K.shape[0] - 1)//2
    dim[1] = (K.shape[1] - 1)//2
    
    N_pad = ( (dim[0], dim[0]), (dim[1], dim[1]) )
    I_pad = np.pad(I, N_pad, 'constant', constant_values = 0)
    
    return toeplitz_m, I_pad.flatten()


def deconv_Wiener(h, i, reg = 0, regularization = 'Tikhonov'):
    
    N, M = i.shape
    
    H, I = convolution_matrix(h, i)
    Ht = np.transpose(H)
    
    if regularization == 'Tikhonov':
        
        R = np.eye(H.shape[1])*reg
        
    elif regularization == 'Laplace':
        
        Z = np.zeros(i.shape)
        Z[N//2, M//2] = 1
        l = laplace(Z)
        L, _ = convolution_matrix(l, i)
        Lt = np.transpose(L)
        R = np.matmul(Lt,L)*reg

    A = inv( np.matmul(Ht, H) + R )
    B = np.matmul(Ht, I)
    OUT =  np.matmul(A,B)

    out = OUT.reshape(N, M)
    
    return out


def deconv_Wiener_FFT(h, i, reg = 0):

    H = np.fft.fft2(h)
    I = np.fft.fft2(i)
    
    A = H*np.conj(H) + reg 
    B = np.conj(H)*I
    OUT =  np.real( np.fft.ifft2(B/A) )
    out = np.fft.fftshift(OUT)
    
    return out


def deconv_RL_FFT(h, i, max_iter = 50, epsilon = None, reg = 0):

    if epsilon is None:
       epsilon = np.finfo(float).eps    

    h = h/np.sum(h) #PSF normalization
    obj = np.ones(i.shape) #Initialization
    
    k = 0
    while k < max_iter:

        conv = convolve( obj, h, mode = 'same' )
        A = np.where(conv < epsilon, 0, i / conv)
        B = convolve( h, A, mode = 'same' )
        C = obj / ( 1 + reg * obj )
        obj = B * C
        
        # flux_obj = np.sum(obj)
        
        # obj *= flux_img / flux_obj
        
        k += 1
    
    return obj