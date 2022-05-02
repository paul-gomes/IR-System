from flask import Flask
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from pre_process import *
import pandas as pd
from flask import jsonify


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
    words =word_tokenize(str(query))
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

def doc_query_similarity(k, query, doc_vector, N, total_vocab, term_df):
    p_query= Pre_Process(query).query_preprocess()
    d_cosines = []
    query_vector = gen_query_vector(p_query, N, total_vocab, term_df)
    for d in doc_vector:
        d_cosines.append(cosine_similarity(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    return out


@app.route('/<string:query>/', methods=['GET'])
def scored_relevent_docs(query):
    print(query)
    data = pd.read_csv('../Data/combined_recipes.csv')
    
    # Load term-df
    with open('../Data/term-df.pickle', 'rb') as handle:
        term_df = pickle.load(handle)
    
    # Load tf-idf
    with open('../Data/tf-idf.pickle', 'rb') as handle:
        tf_idf = pickle.load(handle)
        
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
    
    s_vector = doc_query_similarity(20, query, doc_vector, N, total_vocab, term_df)
    print(s_vector)
    result = []
    for i in s_vector:
        r = {
            'Title': data.at[i,'Title'],
            'Ingredients': data.at[i, 'Ingredients'],
            'Urls':  data.at[i, 'recipe_urls'],
            'Cooking_time': data.at[i, 'Cooking Time']
        }
        result.append(r)
    return jsonify(result)

        
if __name__ == '__main__':
    app.run(port=8000, debug=False)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

