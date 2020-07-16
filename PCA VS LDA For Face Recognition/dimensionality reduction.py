from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

def load_data(path,height,width):
    
    img=os.listdir(path)
    images=np.zeros((width*height,len(img)))
    train_labels=np.zeros(len(img)).astype('uint8')
    for pic,i in zip(img,np.arange(len(img))):
        train_labels[i]=int(pic.split('.')[0][7:9])-1
        image=np.asarray(Image.open(os.path.join(path,pic)).resize((width,height),Image.ANTIALIAS)).flatten()
        images[:,i]=image

    return images,train_labels

def PCA(imgData_train,num_dim=None):
    imgData_mean = np.mean(imgData_train, axis=1).reshape(-1, 1)
    data_centered = imgData_train - imgData_mean
    S = data_centered.T @ data_centered
    eigenvalues, eigenvectors = np.linalg.eig(S)

    #to ordering eigenvectors descending by their eigenvalue indexes
    sort_index = np.argsort(-eigenvalues)
    if num_dim is None:
        for eigenvalue, i in zip(eigenvalues[sort_index], np.arange(len(eigenvalues))):
            if eigenvalue <= 0:
                sort_index = sort_index[:i]
                break
    else:
        sort_index=sort_index[:num_dim]

    
    eigenvalues=eigenvalues[sort_index]
    #transform from data_centered.T @ data_centered value to data_centered@data_centered.T
    eigenvectors=data_centered@eigenvectors[:, sort_index]
    eigenvectors_norm=np.linalg.norm(eigenvectors,axis=0)
    eigenvectors=eigenvectors/eigenvectors_norm

    return eigenvalues,eigenvectors,imgData_mean

def kernel_function(X,gamma,alpha,kernel_type):
    
    sq_dists = pdist(X, 'sqeuclidean')  
    print(sq_dists.shape)
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)
    print(mat_sq_dists.shape)
    # Compute the symmetric kernel matrix.
    if kernel_type == '1':
        K = np.exp(-gamma * mat_sq_dists)
    elif kernel_type == '2':
        K = (1 + gamma*mat_sq_dists/alpha)**(-alpha)
    return K

def kernel_PCA(imgData_train, gamma,alpha, n_components,kernel_type):
   
    # imgData_mean = np.mean(imgData_train, axis=1).reshape(-1, 1)

    K = kernel_function(imgData_train, gamma,alpha,kernel_type)
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)    
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigenvalues, eigenvectors = eigh(K)
    
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1] 
    eigenvectors = np.column_stack([eigenvectors[:, i]
                           for i in range(n_components)])    
    return eigenvectors

def LDA(imgData_train,train_labels,num_dim=None):
    N=imgData_train.shape[0]
    imgData_mean = np.mean(imgData_train, axis=1).reshape(-1, 1)
    Sw = np.zeros((N, N))
    Sb = np.zeros((N, N))
    mean_classes = np.zeros((N, 15))  # 15 classes

    #all classes mean
    for i in range(imgData_train.shape[1]):
        mean_classes[:, train_labels[i]] += imgData_train[:, train_labels[i]]
    mean_classes = mean_classes / 10

    # distance within-class scatter
    for i in range(imgData_train.shape[1]):
        d = imgData_train[:, train_labels[i]].reshape(-1,1) - mean_classes[:, train_labels[i]].reshape(-1,1)
        Sb += d @ d.T

    # distance between-class scatter
    for i in range(15):
        d = mean_classes[:, i].reshape(-1,1) - imgData_mean
        Sw += 10 * d @ d.T

    eigenvalues,eigenvectors=np.linalg.eig(np.linalg.inv(Sb)@Sw)
    sort_index=np.argsort(-eigenvalues)
    if num_dim is None:
        sort_index=sort_index[:-1]  # reduce 1 dim as default setting
    else:
        sort_index=sort_index[:num_dim]

    eigenvalues=np.asarray(eigenvalues[sort_index].real,dtype='float')
    eigenvectors=np.asarray(eigenvectors[:,sort_index].real,dtype='float')

    return eigenvalues,eigenvectors


def kernel_LDA(imgData_train, gamma,alpha, n_components,kernel_type):
   
    # in the MxN dimensional dataset.
    n=imgData_train.shape[0]
    M = np.zeros((n, n))
    N = np.zeros((n, n))
    mean_classes = np.zeros((n, 15))  # 15 classes

    Ms = kernel_function(imgData_train,gamma,alpha,kernel_type)
    #all classes mean
    for i in range(imgData_train.shape[1]):
        mean_classes[:, train_labels[i]] += Ms[:, train_labels[i]]
    mean_classes = mean_classes / 10
    
    imgData_mean = np.mean(Ms, axis=1).reshape(-1, 1)
  
    # distance between-class scatter
    for i in range(15):
        d = mean_classes[:, i].reshape(-1,1) - imgData_mean
        M += 10 * d @ d.T
    
    N = Ms.shape[0]
    one_n = np.ones((N,N)) / N
    I = np.identity(N)
    # distance within-class scatter
    for i in range(imgData_train.shape[1]):
        d = Ms[:, train_labels[i]].reshape(-1,1)*(I-one_n)*Ms[:, train_labels[i]].reshape(-1,1).T
        N += d 
 
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigenvalues, eigenvectors = eigh(np.linalg.pinv(N)@M)
    
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1] 
    eigenvectors = np.column_stack([eigenvectors[:, i]
                           for i in range(n_components)])    
    return eigenvectors
    
def plot_eigenface(imgData_train,n_img,height,width):
    plt.figure(figsize=(8,10))
    plt.subplots_adjust( wspace=0.1 ,hspace=0.4)

    for i in range(n_img):
        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.title(str(i+1) + ' Image')
        plt.imshow(imgData_train[:,i].reshape(height,width),cmap='gist_gray')

    plt.show()

def plt_reconsruct_img(imgData_train,imgData_reconsruct,n_img,height,width):
    plt.figure(figsize=(20,10))
    plt.subplots_adjust( wspace=0.3 ,hspace=0)
    randint=np.random.choice(imgData_train.shape[1],n_img)
    for i in range(n_img):
        plt.subplot(2,n_img,i+1)
        plt.imshow(imgData_train[:,randint[i]].reshape(height,width),cmap='gist_gray')
        plt.axis('off')
        plt.title(str(i+1) + ' Ori Image')

        plt.subplot(2,n_img,i+1+n_img)
        plt.imshow(imgData_reconsruct[:,randint[i]].reshape(height,width),cmap='gist_gray')
        plt.axis('off')
        plt.title(str(i+1) + ' Rec Image')

    plt.show()

def testing_accuracy(imgData_test,test_labels,y_train,train_labels,W,imgData_mean=None,k=1):
    
    if imgData_mean is None:
        imgData_mean=np.zeros((imgData_test.shape[0],1))

    y_test=W.T@(imgData_test-imgData_mean)

    # k-nn classifier
    predicted_labels=np.zeros(y_test.shape[1])
    for i in range(y_test.shape[1]):
        distance=np.zeros(y_train.shape[1])
        for j in range(y_train.shape[1]):
            distance[j]=np.sum(np.square(y_test[:,i]-y_train[:,j]))
        sort_index=np.argsort(distance)
        nearest_neighbors=train_labels[sort_index[:k]]
        unique, counts = np.unique(nearest_neighbors, return_counts=True)
        nearest_neighbors=[k for k,v in sorted(dict(zip(unique, counts)).items(), key=lambda item: -item[1])]
        predicted_labels[i]=nearest_neighbors[0]

    accuracy=np.count_nonzero((test_labels-predicted_labels)==0)/len(test_labels)
    return accuracy


if __name__ == "__main__":
    filepath=('./Yale_Face_Database/Training')
    im = Image.open('./Yale_Face_Database/Training/subject01.centerlight.pgm')
    #conver to grayscale
    im = im.convert("L")
    
    # # resize to given size (if given)
    # get the image height and width
    sz=None
    X=[]
    if (sz is not None):
        im = im.resize(sz, Image.ANTIALIAS)
    X.append(np.asarray(im, dtype=np.uint8))
    height = len(X[0]) #Rows.
    width = len(X[0][1]) #Columns.

    #modify the size
    # height = 80 #Rows.
    # width = 50 #Columns.

    #load_image and make it a 1D vector of each image
    imgData_train,train_labels=load_data(filepath,height,width)

    #number of images to plot the eigenfaces
    n_fig = 25

    #k number of nearest-neighbors
    k=4
    
    gamma = 150
    alpha = 0.1
    n_components=None

    #get the eigenvalues,eigenvectors, and mean of all imgs
    eigenvalues,eigenvectors,imgData_mean=PCA(imgData_train,n_components)
    print(imgData_mean.shape)


    mode=input('Choose a Method \n1: PCA - Eigenfaces, \n2: LDA - Fisherfaces, \n3: Kernel PCA, \n4: kernel LDA: ')
    if mode=='1':
        #eigenvectors as W notation (orthogonal projection)
        W=eigenvectors.copy()
        print('W shape: {}'.format(W.shape))
        
        #plot first n_fig of eigenface
        plot_eigenface(W,n_fig,height,width)

        # principal components of the observed img data train (reduce dimension)
        y=W.T@(imgData_train-imgData_mean)
        print('y shape: {}'.format(y.shape))
        
        # reconstruction image data from the pca basis
        imgData_reconsruct=W@y+imgData_mean

        #plot reconstruction of images from PCA 
        plt_reconsruct_img(imgData_train,imgData_reconsruct,10,height,width)

        # testing the pca model using image data test to get performance of accuracy 
        filepath=os.path.join('Yale_Face_Database','Testing')
        imgData_test,test_labels=load_data(filepath,height,width)
        accuracy=testing_accuracy(imgData_test,test_labels,y,train_labels,W,imgData_mean,k)
        print('Accuracy: {:.2f} %'.format(accuracy*100))

    elif mode=='2':
        # principal components of the observed img data train using the eigenvector from PCA(low dimension)
        imgData_pca=eigenvectors.T@(imgData_train-imgData_mean)

        # get the eigenvalues and eigenvector from LDA subspace
        eigenvalues_lda,eigenvectors_lda=LDA(imgData_pca,train_labels)

        # orthogonal projection of fisherfaces
        W=eigenvectors@eigenvectors_lda
        print('W shape: {}'.format(W.shape))

        #plot first n_fig of eigenface
        plot_eigenface(W,n_fig,height,width)

        # principal components of the observed img data train (low dimension)
        y=W.T@imgData_train
        print('y shape: {}'.format(y.shape))

        # reconstruction image data from the LDA basis
        imgData_reconsruct=W@y+imgData_mean

        #plot reconstruction of images from LDA       
        plt_reconsruct_img(imgData_train,imgData_reconsruct,10,height,width)

        # testing the LDA model using image data test to get performance of accuracy 
        filepath = os.path.join('Yale_Face_Database', 'Testing')
        imgData_test, test_labels = load_data(filepath, height, width)
        accuracy = testing_accuracy(imgData_test, test_labels, y, train_labels, W, imgData_mean,k)
        print('Accuracy: {:.2f}%'.format(accuracy * 100))
    
    elif mode =='3':
        kernel_type=(input('Choose a kernel \n1: Radial basis function, \n2: Quadratic basis function: '))
        W=kernel_PCA(imgData_train,gamma,alpha,n_components,kernel_type=kernel_type)
        print('W shape: {}'.format(W.shape))
        
        # principal components of the observed img data train (low dimension)
        y=W.T@imgData_train
        print('y shape: {}'.format(y.shape))

        # reconstruction image data from the LDA basis
        imgData_reconsruct=W@y+imgData_mean

        #plot reconstruction of images from LDA       
        plt_reconsruct_img(imgData_train,imgData_reconsruct,10,height,width)


        filepath=os.path.join('Yale_Face_Database','Testing')
        imgData_test,test_labels=load_data(filepath,height,width)
        accuracy=testing_accuracy(imgData_test,test_labels,y,train_labels,W,imgData_mean,k)
        print('Accuracy: {:.2f} %'.format(accuracy*100))
    
    elif mode =='4':
        kernel_type=(input('Choose a kernel \n1: Radial basis function, \n2: Quadratic basis function: '))
        # principal components of the observed img data train using the eigenvector from PCA(low dimension)
        A=kernel_LDA(imgData_train,gamma,alpha,n_components,kernel_type=kernel_type)
        print('A shape: {}'.format(A.shape))
        # K =kernel_function(imgData_train,gamma,alpha,kernel_type=kernel_type)
        # K.reshape(133,135)
        # principal components of the observed img data train (low dimension)
        y=A.T@imgData_train
        print('y shape: {}'.format(y.shape))

        # reconstruction image data from the LDA basis
        imgData_reconsruct=A@y+imgData_mean

        #plot reconstruction of images from LDA       
        plt_reconsruct_img(imgData_train,imgData_reconsruct,10,height,width)


        filepath=os.path.join('Yale_Face_Database','Testing')
        imgData_test,test_labels=load_data(filepath,height,width)
        accuracy=testing_accuracy(imgData_test,test_labels,y,train_labels,A,imgData_mean,k)
        print('Accuracy: {:.2f} %'.format(accuracy*100))
    else: print('task is not recognized!')    