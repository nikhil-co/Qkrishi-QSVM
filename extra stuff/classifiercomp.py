import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics
from sklearn.metrics import roc_auc_score

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

bank_data = pd.read_csv('bank_cleaned.csv')
bank_data = bank_data.sample(frac=0.1)

Y = bank_data['y'].values
# X = bank_data[['age','job','marital','education','default','balance',
#                'housing','loan','contact','day','month','duration',
#                'campaign','pdays','previous','poutcome']].values
X = bank_data[['age', 'job', 'marital', 'education', 'default', 'balance',
               'housing', 'loan', 'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome']].values
# Scaling Data
ss = StandardScaler()
X = ss.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.3, random_state=12)

# iterate over classifiers
for name, clf in zip(names, classifiers):

    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    test_metrics = metrics.classification_report(y_true=y_test, y_pred=predictions, zero_division=0)
    rocauc = round(roc_auc_score(y_test, predictions), 4)

    print(f'Testing Metrics for {name}\n{test_metrics}')
    print(f'ROC AUC Score: {rocauc}\n\n')
