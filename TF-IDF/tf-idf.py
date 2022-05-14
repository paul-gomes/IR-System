import pandas as pd
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import pickle
import math
from pre_process import *
import os

nltk.download('stopwords')
nltk.download('punkt')
df_1 = pd.read_csv('../Data/combined_recipes.csv')
df_1 = df_1.drop(columns=['recipe_urls', 'Cooking Time'])
#data = df_1.head(15)
#p_data = pre_process(df_1)

#pre-process data
data_p = Pre_Process(df_1)
p_data = data_p.data_preprocess()


term_df = {}
term_prox = {}
for index, row in p_data.iterrows():
    for col in p_data.columns:
        pos = {}
        words = word_tokenize(str(row[col]))
        for i in range(len(words)): # we need the positions for term proximity
            w = words[i]
            try:
                pos[w].append(i)
            except:
                pos[w] = [i]
        
        for w in words:
            try:
                term_df[w].add(index)
                term_prox[str(index)+"__"+w].add(pos[w])
            except:
                term_df[w] = {index}
                term_prox[str(index)+"__"+w] = pos[w]

with open('../Data/term-df.pickle', 'wb') as handle:
    pickle.dump(term_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../Data/term-prox.pickle', 'wb') as handle:
    pickle.dump(term_prox, handle, protocol=pickle.HIGHEST_PROTOCOL)


def doc_frequency(word):
    try:
        return len(term_df[word])
    except:
        return 0



total_vocab = [x for x in term_df]
N = len(p_data)

#TF-IDF for body/Instructions
tf_idf = {}
for index, row in p_data.iterrows():
    title = word_tokenize(str(row['Title']))
    ingredients = word_tokenize(str(row['Ingredients']))
    instructions = word_tokenize(str(row['Instructions']))
    counter = Counter(title + ingredients + instructions )
    words_count = len(title + ingredients + instructions)

    for w in np.unique(instructions):
        tf = counter[w]
        df = doc_frequency(w)
        idf = np.log((N+1)/(df+1)) #to prevent division by 0
        tf_idf[index, w] = tf*idf


#TF-IDF for ingredients and title
tf_idf_title_ingredient = {}
N = len(p_data)
for index, row in p_data.iterrows():
    title = word_tokenize(str(row['Title']))
    ingredients = word_tokenize(str(row['Ingredients']))
    instructions = word_tokenize(str(row['Instructions']))
    title_ingredients = title + ingredients
    counter = Counter(title + ingredients + instructions )
    words_count = len(title + ingredients + instructions)

    for w in np.unique(title_ingredients):
        tf = counter[w]
        df = doc_frequency(w)
        idf = math.log((N+1)/(df+1)) #to prevent division by 0
        tf_idf_title_ingredient[index, w] = tf*idf


#Assigning more weigts to the words in Title and Ingredients
alpha = 0.8

for key in  tf_idf:
    tf_idf[key] *= alpha

for key in tf_idf_title_ingredient:
    tf_idf[key] = tf_idf_title_ingredient[key]


with open('../Data/tf-idf.pickle', 'wb') as handle:
    pickle.dump(tf_idf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


"""
doc_vector = np.zeros((N, len(total_vocab)))
for i in tf_idf:
    try:
        index = total_vocab.index(i[1])
        doc_vector[i[0]][index] = tf_idf[i]
    except:
        pass




def cosine_similarity(d,q):
    cos_sim = np.dot(d, q)/(np.linalg.norm(d)*np.linalg.norm(q))
    return cos_sim


def gen_query_vector(query):
    words =word_tokenize(str(query))
    query_v = np.zeros((len(total_vocab)))
    counter = Counter(words)
    for w in np.unique(words):
        tf = counter[w]
        df = doc_frequency(w)
        idf = math.log((N+1)/(df+1)) #to prevent division by 0
        try:
            index = total_vocab.index(w)
            query_v[index] = tf*idf
        except:
            pass
    return query_v

def expand_query_vector(query_v, relevant_docs_v, irrelevant_doc):
    alpha = 1
    beta = 0.75
    gamma = 0.15
    avg_r_v = np.mean(relevant_docs_v, axis=0)
    e_query = alpha * query_v + beta * avg_r_v - gamma * irrelevant_doc
    return e_query

def doc_query_similarity(k, query):
    p_query= Pre_Process(query).query_preprocess()

    print("\nQuery:", query)
    print("")

    d_cosines = []
    query_vector = gen_query_vector(p_query)
    for d in doc_vector:
        d_cosines.append(cosine_similarity(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]

    print("")

    print(out)

def doc_query_similarity_with_query_expansion(k, query, relevant_docs_v):
    print(relevant_docs_v)
    print("")
    p_query= Pre_Process(query).query_preprocess()
    d_cosines = []
    query_vector = gen_query_vector(p_query)
    e_query_vector = expand_query_vector(query_vector, relevant_docs_v, doc_vector[-1])
    for d in doc_vector:
        d_cosines.append(cosine_similarity(e_query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    print(out)

query = "beef stew"
r_docs = [377, 760]
doc_query_similarity(10, query)
r_docs = [doc_vector[i] for i in r_docs]
doc_query_similarity_with_query_expansion(10, query, r_docs)

"""










