import numpy as np
import matplotlib.pyplot as plt
import sys

file_name = sys.argv[1]
n = int(input('n-basis: '))
l = int(input('lambda: '))

#length of row in txt file
def file_length(fname):
    with open(fname, 'r') as f:
        num_lines = 0
        for line in f:
            num_lines += 1
        return num_lines

m = file_length(file_name) 

#scan data points
x, y = np.loadtxt(file_name, delimiter=',', usecols=(0,1), unpack=True)

#design X matrix
i1 = np.ones(m)[np.newaxis]
x1 = x[np.newaxis]
xx = np.concatenate((i1.T, x1.T), 1) #X
j = 2

'''n order variable'''
while j<n:
    xx = np.concatenate((xx, (x1 ** j).T), 1)    
    j = j + 1

#design matrix
des_matrix = np.array(np.dot(xx.T, xx)) #X.X^T

#identity matrix
C = np.identity(len(x))

#initialize matrix shape for LSE
lambda_identity = np.zeros_like(des_matrix)
des_lse = np.zeros_like(des_matrix)

'''lambda function - regularization'''
for i in range(0, len(des_matrix)): #diagonal matrix
    for j in range(0, len(des_matrix)):
    
        lambda_identity[i][j] = C[i][j] * l
        des_lse[i][j] = des_matrix[i][j] + lambda_identity[i][j]

#lu decomposition and inverse of matrix
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
        X = np.zeros_like(des_matrix)

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
#help to get value of total error
def get_function_value(coeff, x_p):
        curve = coeff[0]
        for i in range(1, len(coeff)):
                curve = curve + coeff[i]*(x_p ** i)

        return curve

#help to visualization
def get_curve(coeff, x_p):
        max = 0
        min = 0
        for i in range(0, len(x_p)):
                if x[i] > max:
                        max = x_p[i]
                if x[i] < min:
                        min = x_p[i]

        x_axis = np.arange(min, max, 0.01) #to generate points and make a line

        curve = coeff[0]
        for i in range(1, len(coeff)):
                curve = curve + coeff[i]*(x_axis ** i)

        return x_axis, curve


#print out
print('n =', n)
print('lambda = ', l, '\n')


'''LSE METHOD'''
print('LSE:')
#inverse matrix
X_LSE = np.zeros_like(des_lse)
X_LSE = lu_decomposition_inverse(des_lse)

#matrix multiplication, getting coefficients and the fitting line
coeff_lse = np.array(np.dot(np.array(np.dot(X_LSE, xx.T)), y))
str_lse = repr(coeff_lse[0])
for i in range(1, len(coeff_lse)):
        str_lse = str_lse + '+' + repr(coeff_lse[i]) + '*X^' + repr(i) 
print('Fitting line:', str_lse)

#got value from equation respective to the each x value
curve_lse_for_error = get_function_value(coeff_lse, x)

'''LSE equation to get total error'''
#total error = value from predictive equation - actual point T * value from predictive - actual point
total_error_lse = np.array(np.dot((curve_lse_for_error-y).T, (curve_lse_for_error-y)))
print('Total error:', total_error_lse,'\n')

'''NEWTON'S METHOD'''
print('Newtons method:')
#inverse matrix
X_Newt = np.zeros_like(des_matrix)
X_Newt = lu_decomposition_inverse(des_matrix)

iteration = 20
epsilon = 0.0001
niter = 0
f0 = np.array([0 for col in range(len(des_matrix))], dtype=float)[np.newaxis]
y1 = y[np.newaxis]
'''Newtons formula, x1 = xo - inv(X.T * X) * ((X.T * X * xo)-(X.T*Y))'''
while(True):
        err = np.array(np.dot(np.array(np.dot(xx.T, xx)), f0.T)) - np.array(np.dot(xx.T, y1.T)) #((X.T * X * xo)-(X.T*Y))
        if abs(np.sum(err)) < epsilon or niter > iteration:
        # if kol > 2:
                break
        else:
                f1 = f0 - (np.array(np.dot(np.array(lu_decomposition_inverse(des_matrix)), err))).T
        niter = niter + 1
        f0 = f1

coeff_newt = f0[0].T
str_newt = repr(coeff_newt[0])
for i in range(1, len(coeff_newt)):
        str_newt = str_newt + '+' + repr(coeff_newt[i]) + '*X^' + repr(i) 
print('Fitting line:', str_newt)

#total error        
curve_newt_for_error = get_function_value(coeff_newt, x)
total_error_newt = np.array(np.dot((curve_newt_for_error-y).T, (curve_newt_for_error-y)))
print('Total error:', total_error_newt)

#visualisation
ax2 = plt.subplot(211)
ax2.set_title('LSE')
ax2.plot(x, y, 'ro')
x_axis_lse, curve_lse = get_curve(coeff_lse, x)
ax2.plot(x_axis_lse, curve_lse,'black')


ax1 = plt.subplot(212)
ax1.set_title('Newtons method')
ax1.plot(x, y, 'ro')
x_axis_newt, curve_newt = get_curve(coeff_newt, x)
ax1.plot(x_axis_newt, curve_newt, 'black')
plt.subplots_adjust(hspace=0.4)

plt.savefig('output_results.png')
plt.show()
