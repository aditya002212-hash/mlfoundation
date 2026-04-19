# Vectors 
   # 1. cosine similarity  
#      use dot product to find the relation or the cos(theta) between two words bu creating their vector 
#      intitution dot product = 1 (same meaning)
#                 dot product = 0 (orthogonal or no relation)
#                 dot product =-1 (opposite word)
import math
a=[1,2,3]
b=[1,-3,2]


def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    return dot / (mag_a * mag_b)

# normalizing ( making unit vector )
# unit vector help us neglect the magnitude of vector and more focous on the direction
## normalize or making unit vector
def normalize(v):
    mag = math.sqrt(sum(x*x for x in v))
    return [x/mag for x in v]

def cosine_similarity(a, b):
    a = normalize(a)
    b = normalize(b)
    return sum(x*y for x, y in zip(a, b))

v1 = [1, 1]
v2 = [10, 10]

print(cosine_similarity(v1, v2))  # Should be ~1 


# retrive the info 
# in this we make our data store in vectors and our query for which data to be fetch is also vector 
# in retrival process query vector have compute with data vectors and we find their cosine similarity the best result will be fetch 
# same process in rag 

def retrival(query,docs):
  best_score=-1
  best_docs=None
  for doc in docs:
    score=cosine_similarity(query,docs[doc])
    if score>best_score:
      best_score=score
      best_docs=doc
  return best_docs

documents = {
   "doc1": [1, 1],   # math related
   "doc2": [0, 1],   # biology
   "doc3": [-1, 0],  # dance
}
query=[1,1]

result=retrival(query,documents)
print(result)

# matrix 
# in llm we store multiple vectors in matrix and we calculate the process by matrix multiplication
def matrix_multiply(A, B):
    result = []

    for i in range(len(A)):  
        row = []
        for j in range(len(B[0])):
            val = 0
            for k in range(len(B)):
                val += A[i][k] * B[k][j]
            row.append(val)
        result.append(row)

    return result

A = [[1, 2]]
B = [[3], [4]]

print(matrix_multiply(A, B))  # [[11]]

# in llm their are neural layer which help to get the output 
# neural layer help to forward pass the input to get the output 
# neural layer   ( w - weight , x - input , b-bias)
# function 
def neural_layer(w,x,b):   # w is weigth of neuron 
  result=[]                # x is the input 
                           # b is the biasing z=w.x+b
  for i in range(len(w)):

    val=0
    for j in range(len(x)):
      val+=x[j]*w[i][j]
      val+=b[i]
    result.append(val)
  return result

def relu(z):    # activation of the right neural
  return[max(0,val) for val in z]
 # learning how llm work as we set the weight and biasing of each layer 
 # and we take the input and it will compute the output if the prdeicted output is correct then it is right
 # if not the we will find the gradiend descent and then set the adjust the weigths accordingly 
# Input
X = [2, 3]

# Layer 1
W1 = [[1, 2], [3, 4]]
b1 = [1, 1]

# Layer 2
W2 = [[1, 1]]
b2 = [0]

z1 = neural_layer(W1, X, b1)
a1 = relu(z1)

z2 = neural_layer(W2, a1, b2)
y_pred = z2[0]
y_true=22
# loss
loss = (y_pred - y_true)**2

# backward (conceptual)
grad_output = 2 * (y_pred - y_true)

#the learning rate
lr = 0.01

# update weights (simplified)
W2[0][0] -= lr * grad_output * a1[0]
W2[0][1] -= lr * grad_output * a1[1] 


# building neural network 
import random

random.seed(42)

w = [random.random(), random.random()]
b = random.random()
lr = 0.01

data = [
    ([1, 2], 3),
    ([2, 3], 5),
    ([3, 4], 7),
]

for epoch in range(1000):
    total_loss = 0
    for X, y_true in data:
        y_pred = w[0] * X[0] + w[1] * X[1] + b
        loss = (y_pred - y_true) ** 2
        total_loss += loss

        dL_dy = 2 * (y_pred - y_true)
        dw0 = dL_dy * X[0]
        dw1 = dL_dy * X[1]
        db = dL_dy

        w[0] -= lr * dw0
        w[1] -= lr * dw1
        b -= lr * db

    if epoch % 100 == 0:
        print(epoch, total_loss)

print("Weights:", w)
print("Bias:", b)
print("Prediction for [1,2]:", w[0]*1 + w[1]*2 + b)



#in complex neural network we need to go backward and calculating the gradient descent for eack neuron for each layer
# matrix multiplication happen if the columm of first = to row of first and result in row of first into columm of second 
# so in backtracking we need to transpose the ouput matrix 
# transpose shifting the matrix rows into columns and columns into row 
def transpose(a):
  transpose = [[0 for _ in range(len(a))] for _ in range(len(a[0]))]
  for i in range(len(a)):
    for j in range(len(a[0])):
      transpose[j][i] = a[i][j]
  return transpose
a=[[2,3,4],[6,5,7],[7,8,9]]
print(transpose(a))

      