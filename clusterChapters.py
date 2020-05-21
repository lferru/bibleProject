'''
Luis Ferrufino
G#00997076
CS 484-002
Final Project
clusterChapters.py
'''

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_samples = 1184
n_features = 1000
n_components = 1 # number of topics. VERY important parameter that needs tuning
n_top_words = 20
theRecord = [] #records score for parameter-tuning

def print_top_words(model, feature_names, n_top_words):
  
  for topic_idx, topic in enumerate(model.components_):
    
    message = "Topic #%d: " %topic_idx
    message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1: -1]])
    print(message)
  print()

#load in the samples:

data_samples = np.load('./chapters.npy')


#use tf (raw count) features for LDA:

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)

for i in range(20):

  # fit the lda model:

  lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
  lda.fit(tf)
  #print("\nTopics in LDA model:")
  #tf_feature_names = tf_vectorizer.get_feature_names()
  #print_top_words(lda, tf_feature_names, n_top_words)
  score = lda.score(tf)
  theRecord.append(score)
  print("Log likelihood: ", score, "with ", n_components, "topics") #we'd like to maximise this
  print("-->Perplexity: ", lda.perplexity(tf)) #we'd like to minimise this
  n_components += 1

best = np.argmax(theRecord) + 1
print("The best number of topics to use is ", best)
print("\nTopics in the best LDA model:")
lda = LatentDirichletAllocation(n_components=best, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
