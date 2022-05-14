from flask import Flask
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from pre_process import *
import pandas as pd
from flask import jsonify
import os

app = Flask(__name__)

def cosine_similarity(d,q):
    cos_sim = np.dot(d, q)/(np.linalg.norm(d)*np.linalg.norm(q))
    return cos_sim

def doc_frequency(word, term_df):
    try:
        return len(term_df[word])
    except:
        return 0
    
def gen_query_vector(query, N, total_vocab, term_df):
    words = word_tokenize(str(query))
    query_v = np.zeros((len(total_vocab)))
    counter = Counter(words)
    for w in np.unique(words):        
        tf = counter[w]
        df = doc_frequency(w, term_df)
        idf = math.log((N+1)/(df+1)) #to prevent division by 0
        try:
            index = total_vocab.index(w)
            query_v[index] = tf*idf
        except:
            pass
    return query_v

def expand_query_vector(query_v, relevant_docs_v, irrelevant_doc_v):
    alpha = 1
    beta = 0.75
    gamma = 0.15
    avg_r_v = np.mean(relevant_docs_v, axis=0)
    e_query = alpha * query_v + beta * avg_r_v - gamma * irrelevant_doc_v
    return e_query
    
# finds the position of each term in the query, and calculates the average distance between them all
def term_proximity_query(query):
    query = word_tokenize(str(query))

    q_pos = {}
    for i in range(len(query)):
        term = query[i]
        try:
            q_pos[term].append(i) # term may appear multiple times within the query, so we use a list
        except:
            q_pos[term] = [i]

    return term_dists(q_pos)

# takes in the query to know what terms to look for in the doc, and returns the average distance between
# every pair of those terms within the doc
def term_proximity_doc(query,doc_index,term_prox):
    query = word_tokenize(str(query))
    query_set = list(set(query))
    
    # create dictionary of terms containing lists of positions within the document
    # For a document "tasty stew beef corn stew" and a query "beef stew"
    # "beef" : [1]
    # "stew" : [3]
    missing = 0
    d_pos = {}
    for w in query_set:
        try:    # if this query term is in the document
            pos_dict = term_prox[str(doc_index)+"__"+w]
            try:
                for pos in pos_dict:
                    d_pos[w].append(pos)
            except:
                d_pos[w] = [] # term may appear multiple times within the document, so we use a list
                for pos in pos_dict:
                    d_pos[w].append(pos)
                
        except:
            missing += 1
            
    return term_dists(d_pos), missing

# given a list of term positions, find the average distance between all pairs
def term_dists(pos):
    # calculate the distance between terms by taking every pair of terms
    # if a term occurs multiple times, use the minimum distance
    dists = []
    for term1 in pos:                     # for every term
        for term2 in pos:                 # and for every term to compare it with
            min_dist = float("inf")
            for i in range(len(pos[term1])):      # for every time the first term appears
                for j in range(len(pos[term2])):  # for every time the second term appears
                    min_dist = min(abs(pos[term1][i] - pos[term2][j]),min_dist)
            if min_dist != float("inf"):
                dists.append(min_dist)

    dists_avg = 0
    for d in dists:
        dists_avg += d
    dists_avg /= 2          # it goes over every pair twice, we have to correct for that
    if len(dists) != 0:     # if == 0, then empty query and distance doesn't matter
        dists_avg /= (len(dists)/2)

    return dists_avg

def doc_query_similarity(k, query, doc_vector, N, total_vocab, term_df, term_prox):
    p_query= Pre_Process(query).query_preprocess()
    d_cosines = []
    query_vector = gen_query_vector(p_query, N, total_vocab, term_df)

    # calculate how close the terms are within the query
    q_dist = term_proximity_query(p_query)

    for d in range(len(doc_vector)):
        # calculate cosine similarity
        val = cosine_similarity(query_vector, doc_vector[d])

        # calculate how close the query's terms are within this doc
        d_dist, missing = term_proximity_doc(p_query,d,term_prox)
        if d_dist != 0 and missing == 0:    # only add term proximity if the document contains all query terms
            val += q_dist/d_dist *.1

        d_cosines.append(val)

    out = np.array(d_cosines).argsort()[-k:][::-1]
    return out


def doc_query_similarity_with_query_expansion(k, query, doc_vector, N, total_vocab, term_df, term_prox, relevant_docs_v, irrelevant_doc_v):
    p_query= Pre_Process(query).query_preprocess()
    d_cosines = []
    query_vector = gen_query_vector(p_query, N, total_vocab, term_df)
    e_query_vector = expand_query_vector(query_vector, relevant_docs_v, irrelevant_doc_v)

    # calculate how close the terms are within the query
    q_dist = term_proximity_query(p_query)

    for d in range(len(doc_vector)):
        # calculate cosine similarity
        val = cosine_similarity(e_query_vector, doc_vector[d])

        # calculate how close the query's terms are within this doc
        d_dist, missing = term_proximity_doc(p_query,d,term_prox)
        if d_dist != 0 and missing == 0:    # only add term proximity if the document contains all query terms
            val += q_dist/d_dist *.1

        d_cosines.append(val)

    out = np.array(d_cosines).argsort()[-k:][::-1]
    return out


@app.route('/search/<string:query>/', methods=['GET'])
def scored_relevent_docs(query):
    print(query)
    data = pd.read_csv('../Data/combined_recipes.csv')
    
    # Load term-df
    with open('../Data/term-df.pickle', 'rb') as handle:
        term_df = pickle.load(handle)
    
    # Load tf-idf
    with open('../Data/tf-idf.pickle', 'rb') as handle:
        tf_idf = pickle.load(handle)
        
    # Load term-prox
    with open('../Data/term-prox.pickle', 'rb') as handle:
        term_prox = pickle.load(handle)

    total_vocab = [x for x in term_df]
    total_docs = [term_df[x] for x in term_df]
    total_docs = set().union(*total_docs)
    N = len(total_docs)
    

    doc_vector = np.zeros((N, len(total_vocab)))
    for i in tf_idf:
        try:
            index = total_vocab.index(i[1])
            doc_vector[i[0]][index] = tf_idf[i]
        except:
            pass
    
    s_vector = doc_query_similarity(50, query, doc_vector, N, total_vocab, term_df, term_prox)
    print(s_vector)
    result = []
    for i in s_vector:
        r = {
            'DocId': str(i),
            'Title': data.at[i,'Title'],
            'Ingredients': data.at[i, 'Ingredients'],
            'Urls':  data.at[i, 'recipe_urls'],
            'Cooking_time': data.at[i, 'Cooking Time']
        }
        result.append(r)
    return jsonify(result)


@app.route('/expand/<string:query>/<string:irrelevant_doc>/<string:relevant_docs>/', methods=['GET'])
def relevance_feedback(query, irrelevant_doc, relevant_docs):
    print(query)
    print(irrelevant_doc)
    print(relevant_docs)
    relevant_docs = str(relevant_docs).split(",")
    print(relevant_docs)
    data = pd.read_csv('../Data/combined_recipes.csv')
    
    # Load term-df
    with open('../Data/term-df.pickle', 'rb') as handle:
        term_df = pickle.load(handle)
    
    # Load tf-idf
    with open('../Data/tf-idf.pickle', 'rb') as handle:
        tf_idf = pickle.load(handle)
        
    # Load term-prox
    with open('../Data/term-prox.pickle', 'rb') as handle:
        term_prox = pickle.load(handle)

    total_vocab = [x for x in term_df]
    total_docs = [term_df[x] for x in term_df]
    total_docs = set().union(*total_docs)
    N = len(total_docs)
    

    doc_vector = np.zeros((N, len(total_vocab)))
    for i in tf_idf:
        try:
            index = total_vocab.index(i[1])
            doc_vector[i[0]][index] = tf_idf[i]
        except:
            pass
    
    relevant_docs_v = [doc_vector[int(i)] for i in relevant_docs]
    irrelevant_doc_v = doc_vector[int(irrelevant_doc)]
    s_vector = doc_query_similarity_with_query_expansion(50, query, doc_vector, N, total_vocab, term_df, term_prox, relevant_docs_v, irrelevant_doc_v)
    print(s_vector)
    result = []
    for i in s_vector:
        r = {
            'DocId': str(i),
            'Title': data.at[i,'Title'],
            'Ingredients': data.at[i, 'Ingredients'],
            'Urls':  data.at[i, 'recipe_urls'],
            'Cooking_time': data.at[i, 'Cooking Time']
        }
        result.append(r)
    return jsonify(result)
        
if __name__ == '__main__':
    app.run(port=8000, debug=False)
        
"""
# For testing without the application
os.chdir("./TF-IDF")
print(os.getcwd())
try:
    os.chdir("./TF-IDF")
except:
    print(os.getcwd())
scored_relevent_docs("beef stew")
# what gets output is the doc numbers, ranked. In the csv, take the doc num + 2 to find the entry
"""
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

