from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


data_path = "./smsspamcollection/SMSSpamCollection.txt"
input_file = open(data_path, "r")
text = input_file.read()

y =[int(line.split("\t")[0].replace("ham", "0").replace("spam","1")) for line in text.split("\n") if len(line)>0]
texts = [line.split('\t')[1] for line in text.split('\n') if len(line) > 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

from sklearn.cross_validation import cross_val_score
import numpy as np

classifier = LogisticRegression()
print np.mean(cross_val_score(classifier, X, np.array(y), scoring="f1"))

classifier.fit(X, y)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print classifier.predict(vectorizer.transform(["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! Subscribe6GB"]))
print classifier.predict(vectorizer.transform(["Hello, dude! How are you? Let's go travelling ^__^"]))