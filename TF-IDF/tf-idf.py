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
for index, row in p_data.iterrows():
    for col in p_data.columns:
        words = word_tokenize(str(row[col]))
        for w in words:
            try:
                term_df[w].add(index)
            except:
                term_df[w] = {index}

with open('../Data/term-df.pickle', 'wb') as handle:
    pickle.dump(term_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
alpha = 0.5

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


def doc_query_similarity(k, query):
    print("Cosine Similarity")
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



doc_query_similarity(10, "Grilled Chicken")
"""











