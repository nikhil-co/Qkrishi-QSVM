import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score

start = time.time()

bank_data = pd.read_csv('bank_cleaned.csv')
bank_data = bank_data.sample(frac=1, random_state=98)

Y = bank_data['y'].values
# X = bank_data[['age','job','marital','education','default','balance',
#                'housing','loan','contact','day','month','duration',
#                'campaign','pdays','previous','poutcome']].values
X = bank_data[['age', 'job', 'marital', 'education', 'default', 'balance',
               'housing', 'loan', 'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome']].values
# Scaling Data
ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.3, random_state=12)


def classifier():
    model = SVC(kernel='linear', cache_size=5000)

    # Fit the classifier
    model.fit(X_train, Y_train)

    end = time.time()

    # Make predictions
    predictions = model.predict(X_test)

    print(f'Time taken for fitting = {end - start}')

    test_metrics = metrics.classification_report(y_true=Y_test, y_pred=predictions, zero_division=0)
    rocauc = round(roc_auc_score(Y_test, predictions), 4)

    print(test_metrics)
    print('ROC AUC Score:', rocauc)

    confusionMatrix = metrics.confusion_matrix(y_true=Y_test, y_pred=predictions)
    disp = metrics.ConfusionMatrixDisplay(confusionMatrix)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    return None

classifier()
