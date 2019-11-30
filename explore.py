import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from scipy import sparse

#nltk.download('punkt')
#nltk.download('wordnet')



#create a tokenizer used by the vectorizer
class LemmaTokenizer(object):
    #code from aaneloy on kaggle at https://www.kaggle.com/aaneloy
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            return [WordNetLemmatizer().lemmatize(w) for w in word_tokenize(doc)]


train =pd.read_csv("train.tsv",sep='\t')
y_train = train['Sentiment']


y_train_ = y_train.iloc[:78030]
y_test  = y_train.iloc[78030:]

X_train = train.iloc[:78030,:]
X_test = train.iloc[78030:,:]

sns.countplot(y_train)
plt.show()

#create vectrorizer to extract features.
word_vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = None,ngram_range = (1,3), analyzer = 'word', encoding = 'utf-8', tokenizer = LemmaTokenizer())
char_vectorizer =  TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = None,ngram_range = (2,6), analyzer = 'char', encoding = 'utf-8', tokenizer = LemmaTokenizer())

#create document feature matrix for training set
X_train_word = word_vectorizer.fit_transform(X_train['Phrase'])
X_train_char = char_vectorizer.fit_transform(X_train['Phrase'])
#create document feture matrix for testing set
X_test_word = word_vectorizer.transform(X_test['Phrase'])
X_test_char = char_vectorizer.transform(X_test['Phrase'])
#now combine X_train and X_test using a simple row addition
X_train = sparse.hstack([X_train_word,X_train_char])
X_test = sparse.hstack([X_test_word,X_test_char])



print("Train dataset has %d samples and %d features"%X_train.shape)
print("Test dataset has %d samples and %d features"%X_test.shape)
print()

clf = MultinomialNB()

clf.fit(X_train,y_train_)

y_pred = clf.predict(X_test)
print()
print(accuracy_score(y_test, y_pred, normalize=False))
print(accuracy_score(y_test, y_pred, normalize=True))