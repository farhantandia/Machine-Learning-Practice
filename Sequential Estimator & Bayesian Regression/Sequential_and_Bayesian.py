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
    print(z)
    return z

def plot_gaussian(mean,var):
    x=[]
    n=int(input('n data:'))
    for i in range(n):
        x.append(univariate_gaussian_data(mean,var))
    plt.hist(x,30)
    plt.title('Mean:{},and Varinance:{}'.format(mean,var))
    plt.show()

'''Welford's online algorithm'''
def sequential_estimator(mean, var):
    epsilon=1e-4
    n=1
    new_mean=new_var=old_mean =old_var =0
    print('Data new data point source function: N({},{})'.format(mean,var))
    while(True):
        n += 1
        new_x = float(univariate_gaussian_data(mean, var))
        print("Add data new data point: ", new_x)
        new_mean = old_mean + (new_x - old_mean) / n
        new_var= old_var + ((new_x - old_mean)**2)/n -old_var/(n-1) 
        print("Mean = ", new_mean, "Variance = ", new_var)
        if(abs(old_mean - new_mean) < epsilon and abs(old_var - new_var) < epsilon):
            # print('total sampling {} new_data_poin'.format(n))
            break
        old_mean = new_mean
        old_var = new_var

'''Polynomial Basis Linear Model Data Generator'''
def polynomial_basis_data(n, a, w):
    x = random.uniform(-1, 1)
    y=0
    for i in range(n):
        y += w[i] * (x ** i)
    e=univariate_gaussian_data(0,a)
    return x, y + e

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

'''Bayesian Linear Regression'''
def bayesian_linear_regression(b,n,a,w):
    old_mean=np.zeros((n,1))
    old_variance=(1/b)*np.identity(n) #var = 1/b * (I)
    x_point = []
    y_point = []
    mean_data = []
    variance_data = []
    i=1
    k=1000
    while(True):
        new_data_point=polynomial_basis_data(n,a,w)
        new_data_point= tuple([round(x,5) if isinstance(x, float) else x for x in new_data_point]) #round the data points only 5 behind comma 
        print('Add data point {}:\n'.format(new_data_point))

        #make a linear regression coefficient function
        X=np.asarray([pow(new_data_point[0],i) for i in range(n)]).reshape(1,-1)
        y=new_data_point[1]
        
        S=lu_decomposition_inverse(old_variance) #covariance inverse matrix
        new_variance=lu_decomposition_inverse(a*X.T@X+S) 
        new_mean=new_variance@(a*X.T*y+S@old_mean)
        
        np.set_printoptions(precision=10)
        print('Posterior mean: ')
        print(new_mean,'\n')
        print('Posterior variance:')
        print(new_variance,'\n')
        
        #update the parameter (Marginalize distribution)
        predict_mean=np.asscalar(X@new_mean)
        print(predict_mean)
        predict_variance=np.asscalar((1/a)+X@new_variance@X.T)
        print('Predictive distribution ~ N({:.5f},{:.5f})'.format(predict_mean,predict_variance))
        print('-------------------------------------------------')
        
        x_point.append(new_data_point[0])
        y_point.append(new_data_point[1])
        if i==10 or i==50:
            mean_data.append(new_mean)
            variance_data.append(new_variance)

        if mean_rate(old_mean,new_mean)<=0.00001 or i>=k:
            mean_data.append(new_mean)
            variance_data.append(new_variance)
            
            print('iteration:',i)
            print('Delta mean (error rate): {:.5f}'.format(mean_rate(old_mean,new_mean)))
            break

            
        old_mean=new_mean
        old_variance=new_variance
        i+=1

    return x_point, y_point, mean_data, variance_data, i

def mean_rate(new_mean,old_mean): 
    temp = 0 
    for i in range(0, len(new_mean)): 
        temp += ((new_mean[i][0] - old_mean[i][0]) ** 2) 
    return sqrt(temp)

def plot_predictive_distribution(data_points,x_point,y_point,mean_data,variance_data,title):
    mean_line=np.zeros(5000) #how many dot to make a line (higher ->smoother line)
    variance_line=np.zeros(5000)
    n_points=np.linspace(-2,2,5000)

    #calculate the mean & variance for each amount (0,10,50,all) data points
    for i in range(len(n_points)):
        X=np.asarray([pow(n_points[i],k) for k in range(n)])
        mean_line[i]=np.asscalar(X@mean_data)
        variance_line[i]=np.asscalar((a)+X@variance_data@X.T)
        
    plt.plot(x_point[:data_points],y_point[:data_points],'bo')
    plt.plot(n_points,mean_line,'k')
    plt.plot(n_points,mean_line+variance_line,'r')
    plt.plot(n_points,mean_line-variance_line,'r')
    plt.xlim(-2,2)
    plt.ylim(-20,25)
    plt.title(title)
    

if __name__=='__main__':
    
    mode=input('Choose a Task \n(0:Random Data Generator, \n1:Sequential Estimator, \n2:Bayesian Linear Regresion): ')
    if mode=='0':
        print('***Random Data Generator***')
        # n=int(input('Total Data:'))
        m=int(input('mean (m):'))
        v=int(input('variance (s):'))
        gaussian = univariate_gaussian_data(m,v)
        plot_gaussian(m,v)

    elif mode=='1':
        print('***Sequential Estimator***')
        m=int(input('mean (m):'))
        v=int(input('variance (s):'))
        sequential_estimator(m,v)

    elif mode=='2':
        print('***Bayesian Linear Regression***')
        b = float(input('precision(b): '))
        n = int(input('n-basis(n): '))
        a = float(input('variance(a): '))
        w = (input('vector W: ').split(","))
        for i in range(len(w)):
            w[i] = float(w[i])
        x_point, y_point, mean_data, variance_data, i = bayesian_linear_regression(b,n,a,w)
        
        plt.subplots_adjust(hspace=0.4,wspace=0.4)
        plt.subplot(221)
        plot_predictive_distribution(0,x_point,y_point,w,np.zeros((n,n)),'Ground truth')
        plt.subplot(222)
        plot_predictive_distribution(i,x_point,y_point,mean_data[2],variance_data[2],'Predict result') 
        plt.subplot(223)
        plot_predictive_distribution(10,x_point,y_point,mean_data[0],variance_data[0],'10 data incomes') 
        plt.subplot(224)
        plot_predictive_distribution(50,x_point,y_point,mean_data[1],variance_data[1],'50 data incomes')  
        plt.savefig('output_result.png')
        plt.show()
    else:
        raise Exception("Unknown Task, Only have 3 tasks")