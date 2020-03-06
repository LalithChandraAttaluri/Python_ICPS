
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'alt.atheism', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, categories=categories)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
#clf = MultinomialNB()
clf = svm.SVC()
clf.fit(X_train_tfidf, twenty_train.target)


twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)