import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import StandardScaler
import time

start = time.time()

bank_data = pd.read_csv('bank_cleaned.csv')
bank_data = bank_data.sample(n=2000)

y = bank_data['y'].values
X = bank_data[['age','job','marital','education','default','balance',
               'housing','loan','contact','day','month','duration',
               'campaign','pdays','previous','poutcome']].values

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.1, random_state=12)
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

from sklearn.svm import SVC
model = SVC(cache_size=5000)

# Fit the classifier
model.fit(X_train, y_train)

end = time.time()
bank_data = pd.read_csv('bank_cleaned.csv')


y = bank_data['y'].values
X = bank_data[['age','job','marital','education','default','balance',
               'housing','loan','contact','day','month','duration',
               'campaign','pdays','previous','poutcome']].values

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.5, random_state=12)
ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)
# Make predictions
predictions = model.predict(X_test)

print(f'Time taken = {end-start}')

# Calculate metrics
accuracy = accuracy_score(y_test,predictions)
precision = precision_score(y_test,predictions)
recall = recall_score(y_test,predictions)
f1score = f1_score(y_test,predictions)

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.show()

print(f'accuracy = {accuracy}, recall = {recall}, precision = {precision}, f1score = {f1score}')