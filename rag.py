from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
docs=['1. Capital of France - Paris',
'2.  Capital of India - New Delhi',
'3. Capital of Japan - Tokyo',
'4. Capital of Germany - Berlin',
'5. Capital of Italy - Rome']

vector_doc=model.embed_documents(docs)
print(vector_doc)

query='what is capital of germany ?'
vector_query=model.embed_query(query)
print(vector_query)

result=cosine_similarity([vector_query],vector_doc)

print(result)



