from numpy import random, sqrt, log, cos, pi
import numpy as np
import matplotlib.pyplot as plt

'''
UNIVARIATE GAUSSIAN DATA GENERATOR USING BOX-MULLER
'''
def univariate_gaussian_data(mean, var):
    u1 = random.rand()
    u2 = random.rand()
    z = sqrt(-2*log(u1))*cos(2*pi*u2)
    z = z*sqrt(var) + mean
    return z

'''lu decomposition and inverse of matrix'''
def lu_decomposition_inverse(res):
    L = np.zeros_like(res)
    U = np.zeros_like(res)

    #lu decomposition
    for j in range(len(res)):
            U[0][j] = res[0][j]
            L[j][0] = res[j][0]/U[0][0]

    # make the rest value of LU start from row 2 and column 2
    for i in range(1, len(res)):
            for j in range(i, len(res)):
                    s1 = sum((L[i][k1]*U[k1][j]) for k1 in range(0, i))
                    U[i][j] = res[i][j] - s1

                    s2 = sum(L[j][k2]*U[k2][i] for k2 in range(i))
                    L[j][i] = (res[j][i] - s2)/U[i][i]

    #make the base matrix Z(substitution matrix) & X(invers value),
    Z = np.zeros_like(res)
    X = np.zeros_like(res)
    C = np.identity(len(res))

    #forward substitution ,no divided bcs its contain 1 in matrix diagonal
    # Matrix L(3x3) * Matrix Z (3x1) = Matrix B(first column matrix identity, next iteration will use second column)) --> find matrix Z by forward subsitution
    for i in range(0, len(res)):
            for j in range(0, len(res)):
                    s1 = sum((L[i][k1]*Z[k1][j]) for k1 in range(0, i))
                    Z[i][j] = C[i][j] - s1

    #back subs start from index 2->0 with back step -1
    #Matrix U * Matrix X (3x1) = Matrix Z(3x1) --> find matrix X (this is the result of inverse matrix X using LU decomposition) by back subs
    for i in range((len(res)-1), -1, -1):
            for j in range(0, len(res)):
                    s1 = sum((U[i][k1]*X[k1][j]) for k1 in range((i+1), len(res)))
                    X[i][j] = (Z[i][j] - s1)/U[i][i]

    return X

def generate_data_point(mx,my,vx,vy,N):
    
    re=np.empty((N,2))
    for i in range(N):
        re[i,0]=univariate_gaussian_data(mx,vx)
        re[i,1]=univariate_gaussian_data(my,vy)
    return re

def design_matrix(cluster0,cluster1):
    X = np.ones((2 * len(cluster0), 3))
    X[:, 1:] = np.vstack((cluster0, cluster1)) #all row 1 ,stack 2 and 3 column with gaussian data x, y
    return X

def target_class(N):
    Y = np.zeros((2 * N, 1))
    Y[N:] = np.ones((N, 1)) #from row 50 all zeros becomes ones
    return Y

def error_rate(val):
    err = np.sqrt(np.sum(val**2)) #RMSE
    return err

def gradient_descent(X,W,Y,lr):
    i=1
    obj_value=100
    while error_rate(obj_value) >= epsilon :
        obj_value=X.T@((1/(1+np.exp(-X@W)))-Y)
        W=W-lr*obj_value
        i+=1
    return W

def newton_method(X,W,Y,lr):
    N=len(X)
    R=np.zeros((N,N))
    for i in range(N):
        R[i,i]=np.exp(-X[i]@W)/np.power(1+np.exp(-X[i]@W),2)
    H=X.T@R@X
    inv_H=lu_decomposition_inverse(H)
    check_matrix = True if True in np.isnan(np.array(inv_H)) else False
    if check_matrix == True:
        print('Hessian matrix is non-invertible(singular), switch to Gradient descent method')
        return gradient_descent(X,W,Y,lr)
    else: 
        obj_value=100
        while np.sqrt(np.sum(obj_value**2))>=epsilon:
            obj_value=inv_H@X.T@((1/(1+np.exp(-X@W)))-Y)
            W=W-lr*obj_value
        return W

def confusion_matrix(X,Y,W):
        
    N=len(X)
    predict_class=np.empty((N,1))
    for i in range(N):
        predict_class[i]=0 if X[i]@W<0 else 1
    # print(predict_class)
    #compare the class label with predicted class    
    Y_con=np.hstack((Y,predict_class))
    TP=FP=FN=TN=0
    for result in Y_con:
        if result[0]==result[1]==1:
            TP+=1
        elif result[0]==result[1]==0:
            TN+=1
        elif result[0]==1 and result[1]==0:
            FP+=1
        else:
            FN+=1
    table=np.empty((2,2))
    table[0,0]=TP
    table[0,1]=FN
    table[1,0]=FP
    table[1,1]=TN

    Class0_predict=[]
    Class1_predict=[]
    for i in range(N):
        if predict_class[i]==0:
            Class0_predict.append(X[i,1:])
        else:
            Class1_predict.append(X[i,1:])

    return (table,np.array(Class0_predict),np.array(Class1_predict))

def print_result(table):
    print('W:')
    print('{:.10f}'.format(float(W[0])))
    print('{:.10f}'.format(float(W[1])))
    print('{:.10f}'.format(float(W[2])))
    print()
    print('Confusion Matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print('Is cluster 1\t\t{:.0f}\t\t\t{:.0f}'.format(table[0,0],table[0,1]))
    print('Is cluster 2\t\t{:.0f}\t\t\t{:.0f}'.format(table[1,0],table[1,1]))
    print()
    Sensitivity = table[0,0]/(table[0,0]+table[0,1]) #TP/TP+FP
    Specificity = table[1,1]/(table[1,1]+table[1,0]) #TN/TN+FP
    print('Sensitivity (Successfully predict cluster 1): {:.5f}'.format(Sensitivity))
    print('Specificity (Successfully predict cluster 2): {:.5f}'.format(Specificity))

def plot_result(Cluster0,Cluster1,title):
    plt.plot(Cluster0[:,0],Cluster0[:,1],'ro')
    plt.plot(Cluster1[:,0],Cluster1[:,1],'bo')
    plt.title(title)


if __name__ == "__main__":
    
    case=input('Choose a Task \n(1:Case 1, \n2:Case 2, \n3:Custom Case): ')
    if case=='1':
        print('*** Case 1 ***')
        N = 50
        mx1 = my1 = 1
        mx2 = my2 = 10
        vx1 = vy1 = vx2 = vy2 = 2

    elif case=='2':
        print('*** Case 2 ***')
        N = 50
        mx1 = my1 = 1
        mx2 = my2 = 3
        vx1 = vy1 = 2 
        vx2 = vy2 = 4
    else:
        N = int(input("N data: "))
        mx1 = float(input("mx1: "))
        my1 = float(input("my1: "))
        mx2 = float(input("mx2: "))
        my2 = float(input("my2: "))
        vx1 = float(input("vx1: "))
        vy1 = float(input("vy1: "))
        vx2 = float(input("vx2: "))
        vy2 = float(input("vy2: "))

    
    '''Generate  data point based on gaussian distribution'''
    Cluster_0=generate_data_point(mx1,my1,vx1,vy1,N)
    Cluster_1=generate_data_point(mx2,my2,vx2,vy2,N)

    X=design_matrix(Cluster_0,Cluster_1)
    Y=target_class(N)

    '''Gradient descent'''
    epsilon=1e-2
    lr=0.001
    W=np.random.rand(3,1)
    W=gradient_descent(X,W,Y,lr)

    print('\nGradient descent:\n')
    table,Class0_predict,Class1_predict=confusion_matrix(X,Y,W)
    print_result(table)
    
    '''Newtons method'''
    W=np.random.rand(3,1)
    W=newton_method(X,W,Y,lr)

    print('\n----------------------------------------\nNewtons method:\n')
    table,Class0_predict,Class1_predict=confusion_matrix(X,Y,W)
    print_result(table)

    plt.subplots_adjust(hspace=0.4,wspace=0.4)
    plt.subplot(131)
    plot_result(Cluster_0,Cluster_1,'Ground truth')
    plt.subplot(132)
    plot_result(Class0_predict,Class1_predict,'Gradient descent')
    plt.subplot(133)
    plot_result(Class0_predict,Class1_predict,'Newtons method')
    plt.savefig('output_result.png')
    plt.show()
    
