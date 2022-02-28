import numpy as np
import matplotlib.pyplot as plt
# global variable
### using fix variable to simple the procedure of input
is_test = False
file_name = None
n = None
lda = None
data = []

def init():
    global is_test
    global file_name
    global n
    global lda
    if is_test:
        file_name = "a.txt"
        n = 3
        lda = 0
        print("Using file name : ",file_name)
    else:
        print("Input the file path : ")
        file_name = input()
        print("Input n : ")
        n = int(input())
        print("Input lambda : ")
        lda = float(input())
        print("Input info : ",file_name," ,n = ",n," ,lambda = ",lda)
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")
            p = [float(line[0]),float(line[1])]
            data.append(p)

def PolyCoefficients(x, coeffs):
    y = 0
    for i in range(len(coeffs)):
        y += coeffs[i]*x**i
    return y

def print_result(title,L):
    # prepare the text of linear equation
    L = np.reshape(L,L.shape[0])
    text = "Fitting line: "
    for i in range(len(L) - 1, -1 ,-1):
        text = text + str(L[i])
        if i != 0:
            text = text + "X^" + str(i) + " "

    # calc error & prepare point data for ploting
    xPoints = []
    yPoints = []
    err = 0
    for p in data:
        xPoints.append(p[0])
        yPoints.append(p[1])
        y = PolyCoefficients(p[0],L)
        err += ((y - p[1]) ** 2)

    print(title)
    print(text)
    print("Total error: {}".format(err))
    
    # plot data point
    plt.figure()
    plt.title(title.split(":")[0])
    plt.scatter(xPoints, yPoints)
    # plot equation curve
    x = np.linspace(int(min(xPoints)), int(max(xPoints) + 1), 100)
    plt.plot(x, PolyCoefficients(x, L))

# Matrix operation
### Reference : https://www.programiz.com/python-programming/examples/multiply-matrix
def product(A,B):
    if len(B.shape) == 1:
        B = np.reshape(B,(B.shape[0],1))
    result = np.zeros((A.shape[0],B.shape[1]), dtype=np.double)
    # iterate through rows of A
    for i in range(len(A)):
    # iterate through columns of B
        for j in range(len(B[0])):
            # iterate through rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

# implement as np.dot() : https://numpy.org/doc/stable/reference/generated/numpy.dot.html
def dot(A,B):
    if(len(A.shape) == 1 and len(B.shape) == 1):
        result = 0
        for i in range(A.shape[0]):
            result += (A[i] * B[i])
        return result
    else:
        return product(A,B)

### Reference : https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
def LU_factory(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    #Loop over rows
    for i in range(n):
        #Eliminate entries below i with row operations 
        #on U and reverse the row operations to 
        #manipulate L
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
    return L, U

### slove LY = B
def forward_substitution(L,B):
    n = L.shape[0]
    Y = np.zeros_like(B, dtype=np.double)
    #Initializing  with the first row.
    Y[0] = B[0] / L[0, 0]
    for i in range(1, n):
        Y[i] = (B[i] - dot(L[i,:i], Y[:i])) / L[i,i]
    return Y

### slove Y = UX
def back_substitution(U, Y):
    n = U.shape[0]
    X = np.zeros_like(Y, dtype=np.double)

    #Initializing with the last row.
    X[-1] = Y[-1] / U[-1, -1]
    for i in range(n-2, -1, -1):
        X[i] = (Y[i] - dot(U[i,i:], X[i:])) / U[i,i]
    return X

def inverse(A):
    n = A.shape[0]
    B = np.eye(n)
    Ainv = np.zeros((n, n))
    L, U = LU_factory(A)
    for i in range(n):
        Y = forward_substitution(L, B[i, :])
        Ainv[i, :] = back_substitution(U, Y)
    return Ainv

# Linear Regression
def LSE(n,lda):
    # prepare matrix
    X = []
    Y = []
    for x,y in data:
        temp = []
        for i in range(0,n):
            temp.append(x ** i)
        X.append(temp)
        Y.append(y)
    X = np.array(X, dtype=np.double)
    Y = np.array(Y, dtype=np.double)
    ATA = product(X.T,X)
    I = np.identity(ATA.shape[0], dtype=np.double)
    inv_ATA = inverse(ATA + lda * I)
    inv_ATAAT = product(inv_ATA,X.T)
    return product(inv_ATAAT,Y)
    
def Newton(n):
    # prepare matrix
    X = []
    Y = []
    for x,y in data:
        temp = []
        for i in range(0,n):
            temp.append(x ** i)
        X.append(temp)
        Y.append(y)
    X = np.array(X, dtype=np.double)
    Y = np.array(Y, dtype=np.double)
    ATA = product(X.T,X)
    inv_ATA = inverse(ATA)
    inv_ATAAT = product(inv_ATA,X.T)
    return product(inv_ATAAT,Y)

# main
init()
LSE_L = LSE(n,lda)
print_result("LSE:",LSE_L)
New_L = Newton(n)
print_result("Newton's Method: ",New_L)
plt.show()