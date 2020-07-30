import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from numba.errors import NumbaWarning
import warnings, time
warnings.simplefilter('ignore', category=NumbaWarning)

@jit
def load_MNIST():
    data_type = np.dtype("int32").newbyteorder('>') #byte order in big-endian
    print(data_type)
    data = np.fromfile("train-images.idx3-ubyte", dtype = "ubyte")
    X = data[4 * data_type.itemsize:].astype("float64").reshape(60000, 28 * 28)
    X = np.divide(X, 128).astype("int") # divide into two bins
    label = np.fromfile("train-labels.idx1-ubyte",dtype = "ubyte")
    label = label[2 * data_type.itemsize : ].astype("int")
    return (X, label)

@jit
def pix_prob_discrete(train_x,train_y):
    '''
    get pix_prob_discrete conditional on class & dim
    train_x: (60000,784) 0-1 matrix
    train_y: (60000,)
    probability matrix of pixel value==1 (10,784) 
    '''
    labels = np.zeros(10)
    for label in train_y:
        labels[label] += 1

    distribution=np.zeros((10,784))
    for i in range(60000):
        c=train_y[i]
        for j in range(784):
            if train_x[i,j]==1:
                distribution[c,j]+=1

    #normalized
    pix_prob_distribution = distribution / labels.reshape(-1,1)

    return pix_prob_distribution

'''Parameter Initialization'''
@jit
def Parameter_init():
    '''
    lambda is like a prior of each class
    with sum = 1 in (10,1) matrix
    '''
    L=np.random.rand(10)
    L=L/np.sum(L)
    '''
    P[i,j]: pixel value==1 prob in class i's jth feature distribution
    the probability distribution of each class features
    10x784 matrix
    '''
    P = np.random.rand(10,784)
    return L, P

''''Update Posterior'''
@jit
def Expextation_Step(X,L,prob_dist):
    '''
    update posterior using log likelihood
    X_train: (60000,784) 0-1 uint8 matrix
    Lambda: (10,1)
    Distribution: (10,784)
    weight of each class (60000,10)
    '''
    complement_dist=1-prob_dist
    W=np.zeros((60000,10))
    for i in range(60000):
        for j in range(10):
            W[i,j]=np.prod(X[i]*prob_dist[j]+(1-X[i])*complement_dist[j])
    #add prior
    W = W*L.reshape(1,-1)

    #normalized each row to [0,1] & sum=1
    sums = np.sum(W,axis=1).reshape(-1,1)
    sums[sums==0] = 1
    W = W/sums

    return W

'''Update Parameter Lambda and P'''
@jit
def Maximization_Step(A,W):
    '''
    W: (60000,10)
    return: (10,1)
    '''
    L = np.sum(W,axis=0)
    L = L/60000
    '''
    A.T@W -> concate with 1-complement
    A: matrix 60000 x 784
    P = Pixel Distribution of each class (10,784)
    '''
    #normalized W
    sums = np.sum(W,axis=0)
    sums[sums==0] = 1
    P=A.T@(W/sums)
    
    return L.T, P.T

'''Perfect Matching Method'''
@jit
def distribution_matching(GT_distribution,est_distribution):
    '''
    matching GT_distribution to estimate_distribution by minimizing the sum of distance
    gt_dist: (10,784)
    estimate_dist: (10,784)
    return: (10)
    '''
    cost=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            cost[i,j]=np.linalg.norm(GT_distribution[i]-est_distribution[j]) #euclidean distance
    # print(Cost.shape)
    #cost 10x10 matrices
    index_class, classes_order=linear_sum_assignment(cost)

    return classes_order

@jit
def confusion_matrix(label,predict,classes_order):
    for i in range(10):
        c=classes_order[i]
        TP,FN,FP,TN=0,0,0,0
        for i in range(60000):
            if label[i]!=c and predict[i]!=c:
                TN+=1
            elif label[i]==c and predict[i]==c:
                TP+=1
            elif label[i]!=c and predict[i]==c:
                FP+=1
            else:
                FN+=1
        table=np.empty((2,2),dtype=int)
        table[0,0]=TP
        table[0,1]=FN
        table[1,0]=FP
        table[1,1]=TN
        print_result(c,table)

@jit
def print_result(c,table):
    print('------------------------------------------------------------')
    print()
    print('Confusion Matrix {}:'.format(c))
    print('\t\tPredict number {} Predict not number {}'.format(c, c))
    print('Is number {}\t\t{}\t\t{}'.format(c,table[0,0],table[0,1]))
    print('Is not number {}\t\t{}\t\t{}'.format(c,table[1,0],table[1,1]))
    print()
    Sensitivity = table[0,0]/(table[0,0]+table[0,1]) #TP/TP+FP
    Specificity = table[1,1]/(table[1,1]+table[1,0]) #TN/TN+FP
    print('Sensitivity (Successfully predict number {}): {:.5f}'.format(c,Sensitivity))
    print('Specificity (Successfully predict not number {}): {:.5f}'.format(c,Specificity))
    print()

@jit
def print_error_rate(iteration,label,predict,classes_order):
    print('Total iteration to converge: {}'.format(iteration))
    label_value=np.zeros(60000)
    for i in range(60000):
        label_value[i]=classes_order[label[i]]
    error=np.count_nonzero(label_value-predict)
    print('Total error rate: {}'.format(error/60000))

@jit
def convergence_rate(P, new_P):
    delta = np.sum(np.abs(new_P-P))
    return delta

@jit
def plot_mnist(Distribution,classes_order,threshold):
    '''
    plot each classes expected pattern
    with threshold between 0 to 1 (choose the best one)
    '''
    mnist_val=np.asarray(Distribution>threshold,dtype='uint8')
    for i in range(10):
        print('class {}:'.format(i))
        plot_imagination(mnist_val[classes_order[i]])
    return
@jit
def plot_imagination(pattern):
    '''
    :param pattern: (784)
    :return:
    '''
    for i in range(28):
        for j in range(28):
            print(pattern[i*28+j],end=' ')
        print()
    print()
    return

if __name__ == "__main__":
    '''Load the training data'''
    X, y = load_MNIST()
    mnist_pd = pix_prob_discrete(X,y)

    '''Parameter Initialize'''
    L,P = Parameter_init()

    '''Convergence components'''
    epsilon = 0.5
    last_val = 1
    delta_val = 0
    iteration = 1

    '''Program Loop'''
    while True:
        '''E-STEP'''
        W = Expextation_Step(X,L,P)
        '''M-STEP'''
        new_L,new_P = Maximization_Step(X,W)
        '''Convergence rule by difference of P'''
        last_val=delta_val
        delta_val = convergence_rate(P, new_P)

        '''Distribution Matching & Print Imagination'''
        class_order= distribution_matching(mnist_pd,P)
        plot_mnist(P,class_order,threshold=0.4)
        print('No. of Iteration: {}, Difference: {} \n'.format(iteration,delta_val))
        print ('-------------------------------------------------------\n')
        L=new_L
        P=new_P
        iteration+=1

        '''if the difference between new and old data <= epsilon ,BREAK'''
        if abs(last_val-delta_val) <= epsilon:
            '''print all result
            get only the max value of prediction 
            and take into account for confusion matrix
            '''
            maximum_val = np.argmax(W, axis=1) 
            confusion_matrix(y,maximum_val,class_order)
            print_error_rate(iteration,y,maximum_val,class_order)
           
            break





