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

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# %matplotlib inline

# Data of Spam and non spam data from UC Irvine's Spam Repository
data = pd.read_table('SMSSpamCollection',header= None, names = ('outcome', 'content'))
# Read my personal text messages 
new_doc = ["hey dude where are you"]
new_doc_results = ['ham']
external_data = pd.DataFrame({'content':new_doc, 'outcome':new_doc_results})

# Stop words
# nltk.download()
stopwords_set1 = set(nltk.corpus.stopwords.words('english'))
stopwords_set2 = set('for a of the and to in or'.split())
stopwords_set3 = ''
symbol_removed1 = '[^A-Za-z0-9]+'

# 1. Split Data into 
n = 3000
train = data.sample(n, random_state = 111)
test = data[~data.index.isin(train.index)]
validate = test.sample(1000,random_state = 111)
test= test[~test.index.isin(validate.index)]
split_correctly = 0 == sum(validate.index.isin(test.index)) + sum(test.index.isin(train.index)) + sum(validate.index.isin(train.index))
set_n_sizes = 'N\'s in .. train:', train.shape,'test:', test.shape,'validate:', validate.shape
print('Data Split Correct?', split_correctly, '\n'*2, set_n_sizes)

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

def train_model(topic_vec):
    x = pd.DataFrame([dict(row) for row in topic_vec[0]])
    y = (train["outcome"] == "spam").astype(int) 
    lda = LDA()
    mask = np.array([~np.isnan(row).any() for row in x.values])
    x_masked = x[mask]
    y_masked = y[mask]
    lda = lda.fit(x_masked,y_masked)
    return lda,x_masked,y_masked, topic_vec[1],topic_vec[2], topic_vec[3]

def predict_unseen(new_doc_in, stop_words_in, trained_model_in, symbols_to_remove):

    dictionary_in = trained_model_in[3]
    tfidf_in = trained_model_in[4]
    lsi_in = trained_model_in[5]
    lda_in = trained_model_in[0]
    new_doc_in_content = pd.Series(new_doc_in.content)
    new_doc_in_outcome = pd.Series(new_doc_in.outcome)
    
    query = prep_nlp(new_doc_in_content, stop_words_in, symbols_to_remove)
    query_bow = [dictionary_in.doc2bow(corp) for corp in query]
    query_tf = tfidf_in[query_bow] 
    
    x_2 = pd.DataFrame([dict(tf) for tf in lsi_in[query_tf]])
    mask = np.array([~np.isnan(row).any() for row in x_2.values])
    x_2masked = x_2[mask]
    y_2 = (new_doc_in_outcome == "spam").astype(int) 
    
    y_2masked = np.array(y_2[mask])
    x_2masked = lda_in.predict(x_2masked)
    
    return x_2masked,y_2masked

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


train_prepped = prep_nlp(
    data_to_prep = train.content,
    stop_words_in= stopwords_set2,
    symbols_to_remove=symbol_removed1)
built_model = build_model(
    train_data = train_prepped,
    topic_n = 300)
trained_model = train_model(
    topic_vec = built_model)

performance_on_train = performance(
    result_x=trained_model[0].predict(trained_model[1]),
    result_y=np.array(trained_model[2]))
predicted_validate = predict_unseen(
    new_doc_in=validate,
    stop_words_in = stopwords_set2,
    symbols_to_remove = stopwords_set2,
    trained_model_in = trained_model)
performance_on_validate = performance(
    result_x=predicted_validate[0],
    result_y=predicted_validate[1])

    
predicted_external = predict_unseen(
    new_doc_in=external_data,
    stop_words_in = stopwords_set2,
    symbols_to_remove = stopwords_set2,
    trained_model_in = trained_model)

print(predicted_external[0])
# performance_on_external = performance(
#     result_x=predicted_external[0],
#     result_y=predicted_external[1])


# results_out = pd.DataFrame({
#     'Train':performance_on_train,
#     'Validate':performance_on_validate,
#     'external':performance_on_external
# }).set_index(
#     [['Accuracy','Precision','Recall','N Size'],
#      ['% Spam / Ham Correct','% Predicted Spam Actually Spam','% Spam Detected','']])

# print(results_out)