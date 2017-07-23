
# coding: utf-8

# ## SMS Spam Detection for App
# 

# #### Importing Libraries and Modules

# In[13]:

import pandas as pd
import numpy as np 
import warnings
import gensim as gs
import nltk 
from re import sub # import sub to replace items in the followiong list comprehension
from collections import defaultdict
from sklearn.lda import LDA
import matplotlib.pyplot as plt
import re
from scipy.stats import ttest_ind
from sklearn import linear_model
import seaborn as sns; sns.set(color_codes=True)
import pickle

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
get_ipython().magic('matplotlib inline')


# ##### User Input

# In[26]:

new_doc = ["I can probably do something this week? I am almos convinced on idea lol"]
external_data = pd.DataFrame({'content':new_doc})


# In[27]:

# 2. Prep Data
def prep_nlp(data_to_prep, stop_words_in, symbols_to_remove):
    # lower case it
    clean = list(data_to_prep.str.lower())
    # this will tokenize
    clean = [[word for word in document.split()] for document in clean]
    words_to_remove = '|'.join(stop_words_in)
    symbol_remover = '[^A-Za-z0-9]+'
    clean = [[sub(symbol_remover,'',word) for word in text] for text in clean]
    clean = [[sub(words_to_remove,'',word) for word in text] for text in clean]
    return clean


# #### 4. Build unsupervised Model
# 1. Build term frequencies.
# 2. Build dictionray using term frequencies
# 3. Build Corpus /  bag of words 
# 4. Create term frequency inverse document frequency matrix
# 5. Reduce matrix into 300 topics (dimension reduction)

# In[28]:

def build_model(train_data, topic_n):
    frequency = defaultdict(int)
    for text in train_data:
        for token in text:
            frequency[token] += 1
    # get freq > 1
    word_freq_1plus = [[x for x in words if frequency[x] > 1] for words in train_data]
    # Create dictionary
    dictionary = gs.corpora.Dictionary(word_freq_1plus)
    # Create Corpus
    corpus = [dictionary.doc2bow(text) for text in train_data]
    # corpus to tfidf
    tfidf = gs.models.TfidfModel(corpus) 
    corp_tf = tfidf[corpus] 
    # Unsupervised Component. Reduce space into 300 topics. 
    topic_n = topic_n
    lsi = gs.models.LsiModel(corp_tf, id2word=dictionary, num_topics = topic_n)
    corp_topics = lsi[corp_tf] 
    return corp_topics, dictionary, tfidf, lsi    


# #### 5. Build supervised Model
# Take the unsupervised results from LSI (a matrix of 300 topics). Train against outcome variable "Spam" or "Ham" (1 or 0). 
# Use Linear Discriminant Analysis to fit data. 

# In[29]:

def train_model(topic_vec):
    x = pd.DataFrame([dict(row) for row in topic_vec[0]])
    y = (train["outcome"] == "spam").astype(int) 
    lda = LDA()
    mask = np.array([~np.isnan(row).any() for row in x.values])
    x_masked = x[mask]
    y_masked = y[mask]
    lda = lda.fit(x_masked,y_masked)
    return lda,x_masked,y_masked, topic_vec[1],topic_vec[2], topic_vec[3]


# #### 6. Test Model on Unseen Data
# This sets up the function "Predict Unseen" which takes documents which were not in the origial training set. This can be used for validation data, test, or any additional documents. The following steps are applied:
# 1. Run the new documents through the same preparation steps as training. 
# 2. Create bag of words with new data
# 3. Transform into term-frequency, inverse document frequency matrix
# 4. Apply results from latent semantic indexing and remove missing values
# 5. Predict classes based on LSI results into class "Spam" or "Ham" (1 or 0)

# In[30]:

def predict_unseen(new_doc_in, stop_words_in, trained_model_in, symbols_to_remove, y_given = True):

    dictionary_in = trained_model_in[3]
    tfidf_in = trained_model_in[4]
    lsi_in = trained_model_in[5]
    lda_in = trained_model_in[0]
    new_doc_in_content = pd.Series(new_doc_in.content)
    
    
    query = prep_nlp(new_doc_in_content, stop_words_in, symbols_to_remove)
    query_bow = [dictionary_in.doc2bow(corp) for corp in query]
    query_tf = tfidf_in[query_bow] 
    
    x_2 = pd.DataFrame([dict(tf) for tf in lsi_in[query_tf]])
    
    if y_given == True: 
        new_doc_in_outcome = pd.Series(new_doc_in.outcome)
        mask = np.array([~np.isnan(row).any() for row in x_2.values])
        x_2masked = x_2[mask]
        y_2 = (new_doc_in_outcome == "spam").astype(int) 
        y_2masked = np.array(y_2[mask])
        x_2masked = lda_in.predict(x_2masked)
        return x_2masked,y_2masked
    else:
        return lda_in.predict(x_2)
        


# ##### 7.Performance
# There are three performance metrics:
# 1. "Accuracy" which tells us, what percent of predicted results equal the actual results
# 2. "Precision": Of all all observations we predicted as spam, what is actually spam?
# 3. "Recall": Of all observations actually spam, what percent did we predict?

# In[31]:

def performance(result_x, result_y):
    actual_positive = result_y == 1
    actual_negative = result_y ==0
    true_positives = result_x[actual_positive] == 1
    false_positives = result_x[actual_negative] == 1
    true_negatives = result_x[actual_negative] == 0
    false_negatives = result_x[actual_positive] == 0
    #A. Accuracy = (TP + TN)/(TP + TN + FP + FN)
    #B. Precision = TP/(TP + FP)
    #C. Recall = TP/(TP + FN)
    accuracy = sum((result_x == result_y))/len(result_y)
    precision = sum(true_positives) / (sum(true_positives) + sum(false_positives))
    recall = sum(true_positives) / (sum(true_positives) + sum(false_negatives))
    return [accuracy, precision, recall, len(result_x)]


# #### Train Model

# In[32]:

retrain = False
if retrain == True:
        # 1. Split Data into 
    n = 3000
    train = data.sample(n, random_state = 111)
    test = data[~data.index.isin(train.index)]
    validate = test.sample(1000,random_state = 111)
    test= test[~test.index.isin(validate.index)]
    split_correctly = 0 == sum(validate.index.isin(test.index)) + sum(test.index.isin(train.index)) + sum(validate.index.isin(train.index))
    set_n_sizes = 'N\'s in .. train:', train.shape,'test:', test.shape,'validate:', validate.shape
    print('Data Split Correct?', split_correctly, '\n'*2, set_n_sizes)
    data = pd.read_table('SMSSpamCollection',header= None, names = ('outcome', 'content'))
    stopwords_set1 = set(nltk.corpus.stopwords.words('english'))
    stopwords_set2 = set('for a of the and to in or'.split())
    stopwords_set3 = ''
    symbol_removed1 = '[^A-Za-z0-9]+'
    train_prepped = prep_nlp(data_to_prep = train.content,stop_words_in= stopwords_set2,symbols_to_remove=symbol_removed1)
    built_model = build_model(train_data = train_prepped,topic_n = 300)
    trained_model = train_model(topic_vec = built_model)
    pickle.dump(trained_model, open( "trained.p", "wb" ) )
else:
    trained_model = pickle.load( open( "trained.p", "rb" ) )        


# #### Execute user input

# In[33]:

predicted_external = predict_unseen(
    new_doc_in=external_data,
    stop_words_in = stopwords_set2,
    symbols_to_remove = stopwords_set2,
    trained_model_in = trained_model,y_given = False
)
predicted_external

