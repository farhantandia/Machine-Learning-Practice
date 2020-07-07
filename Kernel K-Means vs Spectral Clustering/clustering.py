import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import sys
from scipy.spatial.distance import squareform, pdist
from array2gif import write_gif

def load_data(path):
    image = cv2.imread(path)
    height, width, channel = image.shape
    img_array = np.zeros((width * height, channel))
    for i in range(height):
        img_array[i * width:(i + 1) * width] = image[i]

    return img_array,height,width

def kernel_function(img_array,gamma_s,gamma_c):
    
    n=len(img_array)
    Spatial_in=np.zeros((n,2))
    for i in range(n):
        Spatial_in[i]=[i//100,i%100]
    K=squareform(np.exp(-gamma_s*pdist(Spatial_in,'sqeuclidean')))*squareform(np.exp(-gamma_c*pdist(img_array,'sqeuclidean')))

    return K

def initial_centroid(img_array,k,centroid_method):
    
    centroids = np.zeros((k, img_array.shape[1]))
    if centroid_method == 'kmeans++':
        centroids[0]=img_array[np.random.randint(low=0,high=img_array.shape[0],size=1),:]

        for c_id in range(1,k): 
            ## initialize a list to store distances of img_array 
            ## points from nearest centroid 
            temp_dist=np.zeros((len(img_array),c_id))
            for i in range(len(img_array)):
                for j in range(c_id):
                    temp_dist[i,j]=np.sqrt(np.sum((img_array[i]-centroids[j])**2))
                dist=np.min(temp_dist,axis=1)
                sum=np.sum(dist)*np.random.rand()
                for i in range(len(img_array)):
                    sum-=dist[i]
                    if sum<=0:
                        centroids[c_id]=img_array[i]
                        break


    elif centroid_method=='random':
        X_mean=np.mean(img_array,axis=0)
        X_std=np.std(img_array,axis=0)
        for c in range(img_array.shape[1]):
           centroids[:,c]=np.random.normal(X_mean[c],X_std[c],size=k)
        
    else:
        raise Exception ('unavailable centroid method!')

    return centroids

def kmeans(img_array,k,H,W,centroid_method):
  
    init_mean=initial_centroid(img_array,k,centroid_method)

    class_cluster=np.zeros(len(img_array),dtype=np.uint8)
    color_map=[]
    n_iteration=1
    while True :
        # Expectation-step
        for i in range(len(img_array)):
            euc_distance=[]
            for j in range(k):
                euc_distance.append(np.sqrt(np.sum((img_array[i]-init_mean[j])**2)))
            class_cluster[i]=np.argmin(euc_distance)

        # Maximization-step
        New_Mean=np.zeros(init_mean.shape)
        for i in range(k):
            assign=np.argwhere(class_cluster==i).reshape(-1)
            for j in assign:
                New_Mean[i]=New_Mean[i]+img_array[j]
            if len(assign)>0:
                New_Mean[i]=New_Mean[i]/len(assign)

        diff = np.sum((New_Mean - init_mean)**2)
        init_mean=New_Mean
        assign_color = cluster_color(class_cluster,k,H,W)
        color_map.append(assign_color)
        
        print('Iteration {}'.format(n_iteration))
        for i in range(k):
            print('k={} : (N pixel assigned : {})'.format(i + 1, np.count_nonzero(class_cluster == i)))
        print('Difference :{}'.format(diff))
        print()
        cv2.imshow('', assign_color)
        cv2.waitKey(1)

        n_iteration+=1
        if diff<EPS:
            break
    return class_cluster,color_map,n_iteration

def cluster_color(img_array,k,H,W):
    
    if k >=5:
         cluster_color= np.random.choice(range(256),size=(k,3))
    elif k ==2 :
        cluster_color = np.array([[0,255,0],[0,0,255]])
    elif k == 3:
        cluster_color = np.array([[255,0,0],[0,255,0],[0,0,255]])
    elif k == 4:
        cluster_color = np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0]])
    
    res=np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            res[h,w,:]=cluster_color[img_array[h*W+w]]
    return res.astype(np.uint8)

def plot3D_eigenspace_laplacian(x_axis,y_axis,z_axis,class_cluster):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    markers=['o','^','s']
    for marker,i in zip(markers,np.arange(3)):
        ax.scatter(x_axis[class_cluster==i],y_axis[class_cluster==i],z_axis[class_cluster==i],marker=marker)
    plt.title('3D representation of Eigenspace coordinates')
    ax.set_xlabel('1st Eigenvector')
    ax.set_ylabel('2nd Eigenvector')
    ax.set_zlabel('3rd Eigenvector')
    plt.show()

def plot2D_eigenspace_laplacian(x_axis,y_axis,z_axis,class_cluster):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    colors=['blue','orange']
    for colors,i in zip(colors,np.arange(2)):
        ax.scatter(x_axis[class_cluster==i],y_axis[class_cluster==i],c=colors,s=3)
    plt.title('2D representation of Eigenspace coordinates')
    ax.set_xlabel('1st Eigenvector')
    ax.set_ylabel('2nd Eigenvector')
    plt.show()

def process_gif(color_map,gif_path):
    for i in range(len(color_map)):
        color_map[i] = color_map[i].transpose(1, 0, 2)
    write_gif(color_map, gif_path, fps=2)

if __name__ == "__main__":
       
    print('Choose image (1 or 2) :')
    inp = int(input())
    print('set n cluster:')
    k = int(input())
    
    print('set gamma spatial similarity:')
    gamma_s=float(input())
    
    print('set gamma color similarity:')
    gamma_c=float(input())

    file_path='image'+str(inp)+'.png'
    img_array,height,width=load_data(file_path)

    initCentroid_method='kmeans++'

    # # # k=3  # k clusters
    # gamma_c =0.00007
    # gamma_s =0.00007
    
    EPS=1e-8
    mode=input('Choose a Clustering Method \n(1:Kernel K-means, \n2:Normalized Spectral Clustering, \n3:Ratio Cut Spectral Clustering): ')
    if mode=='1':
        start = np.around(time.time(), decimals=0)
        print('Clustering the image...')
        Gram=kernel_function(img_array,gamma_s,gamma_c)
        assigned_pixel,color_map,n_iteration=kmeans(Gram,k,height,width,centroid_method=initCentroid_method)
                
        newTime = np.around(time.time(), decimals=0)
        print("time needed for eigen decomposition: ", newTime-start, " s")
        save_path=os.path.join('{}_{}_{}_gammaS{}_gammaC{}_k{}_{}'.format(file_path.split('.')[0],initCentroid_method,n_iteration-1,gamma_s,gamma_c,k,'kernel k-means.gif'))
        process_gif(color_map,save_path)
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode=='2':
        start = np.around(time.time(), decimals=0)
        print('Clustering the image...')
        # similarity matrix
        W=kernel_function(img_array,gamma_s,gamma_c)
        # degree matrix
        D=np.diag(np.sum(W,axis=1))
        L=D-W
        D_inverse_square_root=np.diag(1/np.diag(np.sqrt(D)))
        L_sym=D_inverse_square_root@L@D_inverse_square_root

        eigenvalue,eigenvector=np.linalg.eig(L_sym)
        sort_index=np.argsort(eigenvalue)
        # U:(n,k)
        U=eigenvector[:,sort_index[1:1+k]]
        # T:(n,k) each row with norm 1
        sums=np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
        T=U/sums

        # k-means
        assigned_pixel,color_map,n_iteration=kmeans(T,k,height,width,centroid_method=initCentroid_method)
        
        newTime = np.around(time.time(), decimals=0)
        print("time needed for eigen decomposition: ", newTime-start, " s")
        save_path=os.path.join('{}_{}_{}_gammaS{}_gammaC{}_k{}_{}'.format(file_path.split('.')[0],initCentroid_method,n_iteration-1,gamma_s,gamma_c,k,'Normalized_Spectral_Cls.gif'))
        process_gif(color_map,save_path)
        if k==3:
            plot3D_eigenspace_laplacian(U[:,0],U[:,1],U[:,2],assigned_pixel)
        elif k==2:
            plot2D_eigenspace_laplacian(U[:,0],U[:,1],0,assigned_pixel)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode=='3':
        
        start = np.around(time.time(), decimals=0)
        print('Clustering the image...')

        # similarity matrix
        W=kernel_function(img_array,gamma_s,gamma_c)
        # degree matrix
        D=np.diag(np.sum(W,axis=1))
        #laplacian grah matrix
        L=D-W
        eigenvalue,eigenvector=np.linalg.eig(L)
        sort_index=np.argsort(eigenvalue)
        # U = eigenvector matrix nxk
        U=eigenvector[:,sort_index[1:1+k]]

        # k-means
        assigned_pixel,color_map,n_iteration=kmeans(U,k,height,width,centroid_method=initCentroid_method)

        newTime = np.around(time.time(), decimals=0)
        print("time needed for eigen decomposition: ", newTime-start, " s")
        save_path=os.path.join('{}_{}_{}_gammaS{}_gammaC{}_k{}_{}'.format(file_path.split('.')[0],initCentroid_method,n_iteration-1,gamma_s,gamma_c,k,'RatioCut_Spectral_Cls.gif'))
        process_gif(color_map,save_path)
        
        if k==3 :
            plot3D_eigenspace_laplacian(U[:,0],U[:,1],U[:,2],assigned_pixel)
        elif k==2:
            plot2D_eigenspace_laplacian(U[:,0],U[:,1],0,assigned_pixel)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else: print('task is not recognized!')