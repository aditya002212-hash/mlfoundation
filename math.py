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