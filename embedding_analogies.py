# TODO: This project is currently onhold. There is not significant enough data to compute the PCA reduction to the matching dimensionality of GLOVE-50
# In addition, the Glove vectors probably dont have as much information encoded in and there is no guarantee of sufficient performance.
# At any rate, it would be best if using a large vector database of precomputed embeddings for GPT. I'm surprised one is not easily accessible yet.

import openai
import numpy as np

import os

openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    raise Exception('OPENAI_API_KEY not found in the environment variables.')


openai.api_key = openai_api_key


# Get embeddings for the words
response_empathy = openai.Embedding.create(
    input="Empathy",
    model="text-embedding-ada-002"
)
embedding_empathy = response_empathy['data'][0]['embedding']

response_feelings = openai.Embedding.create(
    input="Feelings",
    model="text-embedding-ada-002"
)
embedding_feelings = response_feelings['data'][0]['embedding']

response_compassion = openai.Embedding.create(
    input="Compassion",
    model="text-embedding-ada-002"
)
embedding_compassion = response_compassion['data'][0]['embedding']

# Perform the vector operation
result_embedding = np.add(embedding_empathy, embedding_feelings) - embedding_compassion


""" USING WORD2VEC FOR LOOKUP """
import gensim.downloader as api
import os

info = api.info()  # show info about available models/datasets
model = api.load("glove-twitter-50")  # download the model and return as object ready for use

from sklearn.decomposition import PCA

# Initialize PCA - set the number of components to 50
pca = PCA(n_components=50)

def get_word_from_embedding(embedding):
    # Normalize the embedding to unit length
    embedding = embedding / np.linalg.norm(embedding)

    # Reshape the embedding to 2D since PCA in sklearn does not support 1D arrays
    embedding = embedding.reshape(1, -1)

    # Apply PCA to reduce the dimensionality of the embedding
    reduced_embedding = pca.fit_transform(embedding)

    # Get the most similar word to the given reduced embedding
    most_similar = model.most_similar(positive=[reduced_embedding[0]], topn=1)
    
    # Return just the word of the most similar result
    return most_similar[0][0]


result_word = get_word_from_embedding(result_embedding)
print(result_word)
