import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.spatial.distance import pdist,squareform
from mpl_toolkits.mplot3d import Axes3D
from array2gif import write_gif


def initial_mean(X,k,initType):
    '''
    @param X: (#datapoint,#features) ndarray
    @param k: #clusters
    @param initType: 'random','pick','k_means_plusplus'
    @return: (k,#features) ndarray, Kij: cluster i's j-dim value
    '''
    Cluster = np.zeros((k, X.shape[1]))
    if initType == 'k_means_plusplus':
        # reference: https://www.letiantian.me/2014-03-15-kmeans-kmeans-plus-plus/
        #pick 1 cluster_mean
        Cluster[0]=X[np.random.randint(low=0,high=X.shape[0],size=1),:]
        #pick k-1 cluster_mean
        for c in range(1,k):
            Dist=np.zeros((len(X),c))
            for i in range(len(X)):
                for j in range(c):
                    Dist[i,j]=np.sqrt(np.sum((X[i]-Cluster[j])**2))
            Dist_min=np.min(Dist,axis=1)
            sum=np.sum(Dist_min)*np.random.rand()
            for i in range(len(X)):
                sum-=Dist_min[i]
                if sum<=0:
                    Cluster[c]=X[i]
                    break
    elif initType=='pick':
        random_pick=np.random.randint(low=0,high=X.shape[0],size=k)
        Cluster=X[random_pick,:]
    else: # initType=='random'
        X_mean=np.mean(X,axis=0)
        X_std=np.std(X,axis=0)
        for c in range(X.shape[1]):
            Cluster[:,c]=np.random.normal(X_mean[c],X_std[c],size=k)

    return Cluster

def kmeans(X,k,H,W,initType='random',gifPath='default.gif'):
    '''
    k clusters
    @param X: (#datapoint,#features) ndarray
    @param k: # clusters
    @param H: image H
    @param W: image W
    @return: (#datapoint) ndarray, Ci: belonging class of each data point
    @return: ndarray list ready for gif
    '''
    Mean=initial_mean(X,k,initType)

    # Classes of each Xi
    C=np.zeros(len(X),dtype=np.uint8)
    segments=[]

    diff=1e9
    count=1
    while diff>EPS :
        # E-step
        for i in range(len(X)):
            dist=[]
            for j in range(k):
                dist.append(np.sqrt(np.sum((X[i]-Mean[j])**2)))
            C[i]=np.argmin(dist)

        # M-step
        New_Mean=np.zeros(Mean.shape)
        for i in range(k):
            belong=np.argwhere(C==i).reshape(-1)
            for j in belong:
                New_Mean[i]=New_Mean[i]+X[j]
            if len(belong)>0:
                New_Mean[i]=New_Mean[i]/len(belong)

        diff = np.sum((New_Mean - Mean)**2)
        Mean=New_Mean

        # visualize
        segment = visualize(C,k,H,W)
        segments.append(segment)
        print('iteration {}'.format(count))
        for i in range(k):
            print('k={}: {}'.format(i + 1, np.count_nonzero(C == i)))
        print('diff {}'.format(diff))
        print('-------------------')
        cv2.imshow('', segment)
        cv2.waitKey(1)

        count+=1

    return C,segments

def load_data(path):
    '''
    @param path:
    @return: (H*W,C) flatten_image ndarray
    '''
    image = cv2.imread(path)
    H, W, C = image.shape
    image_flat = np.zeros((W * H, C))
    for h in range(H):
        image_flat[h * W:(h + 1) * W] = image[h]

    return image_flat,H,W

def precomputed_kernel(X,gamma_s,gamma_c):
    '''
    kernel function: k(x,x')= exp(-r_s*||S(x)-S(x')||**2)* exp(-r_c*||C(x)-C(x')||**2)
    :@param X: (H*W=10000,rgb=3) ndarray
    :@param gamma_s: gamma of spacial
    :@param gamma_c: gamma of color
    :@return : (10000,10000) ndarray
    '''
    n=len(X)
    # S(x) spacial ingormation
    S=np.zeros((n,2))
    for i in range(n):
        S[i]=[i//100,i%100]
    K=squareform(np.exp(-gamma_s*pdist(S,'sqeuclidean')))*squareform(np.exp(-gamma_c*pdist(X,'sqeuclidean')))

    return K

def visualize(X,k,H,W):
    '''
    @param X: (10000) belonging classes ndarray
    @param k: #clusters
    @param H: image_H
    @param W: image_W
    @return : (H,W,3) ndarray
    '''
    colors= colormap[:k,:]
    res=np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            res[h,w,:]=colors[X[h*W+w]]

    return res.astype(np.uint8)

def plot_eigenvector(xs,ys,zs,C):
    '''
    only for 3-dim datas
    @param xs: (#datapoint) ndarray
    @param ys: (#datapoint) ndarray
    @param zs: (#datapoint) ndarray
    @param C: (#datapoint) ndarray, belonging class
    '''
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    markers=['o','^','s']
    for marker,i in zip(markers,np.arange(3)):
        ax.scatter(xs[C==i],ys[C==i],zs[C==i],marker=marker)

    ax.set_xlabel('eigenvector 1st dim')
    ax.set_ylabel('eigenvector 2nd dim')
    ax.set_zlabel('eigenvector 3rd dim')
    plt.show()

def save_gif(segments,gif_path):
    for i in range(len(segments)):
        segments[i] = segments[i].transpose(1, 0, 2)
    write_gif(segments, gif_path, fps=2)

if __name__ == "__main__":
        # predefine colormap
    colormap= np.random.choice(range(256),size=(100,3))
    #colormap=np.asarray([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[255,255,255]],dtype=np.uint8)

    EPS=1e-9
    # set parameters
    img_path='image1.png'
    image_flat,HEIGHT,WIDTH=load_data(img_path)
    gamma_s=0.001
    gamma_c=0.001
    k=3  # k clusters
    k_means_initType='k_means_plusplus'
    gif_path=os.path.join('GIF','{}_{}Clusters_{}'.format(img_path.split('.')[0],k,'kernel k-means.gif'))




    mode=input('Choose a Task \n(1:Kernel K-means, \n2:Normalized Spectral Clustering, \n3:Unnormalized Spectral Clustering): ')
    if mode=='1':
	    
        Gram=precomputed_kernel(image_flat,gamma_s,gamma_c)
        belongings,segments=kmeans(Gram,k,HEIGHT,WIDTH,initType=k_means_initType)
        save_gif(segments,gif_path)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode=='2':
        # similarity matrix
        W=precomputed_kernel(image_flat,gamma_s,gamma_c)
        # degree matrix
        D=np.diag(np.sum(W,axis=1))
        L=D-W
        D_inverse_square_root=np.diag(1/np.diag(np.sqrt(D)))
        L_sym=D_inverse_square_root@L@D_inverse_square_root

        eigenvalue,eigenvector=np.linalg.eig(L_sym)
        np.save('{}_eigenvalue_{:.3f}_{:.3f}_normalized'.format(img_path.split('.')[0],gamma_s,gamma_c),eigenvalue)
        np.save('{}_eigenvector_{:.3f}_{:.3f}_normalized'.format(img_path.split('.')[0],gamma_s,gamma_c),eigenvector)
        

        eigenvalue=np.load('{}_eigenvalue_{:.3f}_{:.3f}_normalized.npy'.format(img_path.split('.')[0],gamma_s,gamma_c))
        eigenvector=np.load('{}_eigenvector_{:.3f}_{:.3f}_normalized.npy'.format(img_path.split('.')[0],gamma_s,gamma_c))
        sort_index=np.argsort(eigenvalue)
        # U:(n,k)
        U=eigenvector[:,sort_index[1:1+k]]
        # T:(n,k) each row with norm 1
        sums=np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
        T=U/sums

        # k-means
        belonging,segments=kmeans(T,k,HEIGHT,WIDTH,initType=k_means_initType)

        save_gif(segments,gif_path)
        if k==3:
            plot_eigenvector(U[:,0],U[:,1],U[:,2],belonging)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode=='3':
        
        # similarity matrix
        W=precomputed_kernel(image_flat,gamma_s,gamma_c)
        # degree matrix
        D=np.diag(np.sum(W,axis=1))
        L=D-W
        
        eigenvalue,eigenvector=np.linalg.eig(L)
        np.save('{}_eigenvalue_{:.3f}_{:.3f}_unnormalized'.format(img_path.split('.')[0],gamma_s,gamma_c),eigenvalue)
        np.save('{}_eigenvector_{:.3f}_{:.3f}_unnormalized'.format(img_path.split('.')[0],gamma_s,gamma_c),eigenvector)
        
        eigenvalue=np.load('{}_eigenvalue_{:.3f}_{:.3f}_unnormalized.npy'.format(img_path.split('.')[0],gamma_s,gamma_c))
        eigenvector=np.load('{}_eigenvector_{:.3f}_{:.3f}_unnormalized.npy'.format(img_path.split('.')[0],gamma_s,gamma_c))
        sort_index=np.argsort(eigenvalue)
        # U
        U=eigenvector[:,sort_index[1:1+k]]

        # k-means
        belonging,segments=kmeans(U,k,HEIGHT,WIDTH,initType=k_means_initType)

        save_gif(segments,gif_path)
        if k==3:
            plot_eigenvector(U[:,0],U[:,1],U[:,2],belonging)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else: print('task is not recognized!')