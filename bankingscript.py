import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import qsvm
import fitness
import gsvm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score


nqubits = 3
depth = 3
output="bank_testdata.csv"
df = pd.read_csv('bank_cleaned.csv')

bank_data = df.sample(n=20000)#,random_state=1)
#bank_data = np.around(bank_data)

y = bank_data['y'].values
X = bank_data[['age','job','marital','education','default','balance','housing','loan','contact',
                'day','month','duration','campaign','pdays','previous','poutcome']].values
start = time.time()

pop, pareto, logbook = gsvm.gsvm(nqubits=nqubits, depth=depth, nparameters=2,
                                    X=X, y=y, weights=[-1.0,1.0],
                                    mu=10,lambda_=5,ngen=5,mutpb=0.25,cxpb=.75)
sim_time = time.time()

print(f'Simulation finished after {sim_time-start} seconds')
print('---------------------------------------------')

with open(output, "w") as f:
    for ide, ind in enumerate(pareto):
        genes=''.join(str(i) for i in list(ind))
        gates, acc = ind.fitness.values
        line = f'{ide},"{genes}",{gates},{acc}'
        f.write(line)
        f.write('\n')
        #print(line)

iot_result = pd.read_csv('bank_testdata.csv',header=None)

def ordenar_salidas_pareto(dataframe):
    dataframe.columns=['ind','circ','gates','accuracy']
    dataframe.sort_values(['gates','accuracy'], ascending=[False,False],inplace=True)
    dataframe.reset_index(inplace=True)
    dataframe.pop('index')
    return dataframe

iot_salidas = ordenar_salidas_pareto(iot_result)

def featuremap_performance(pop:str,nqubits:int) -> None:
    '''Returns the performance of a feature map on all of the dataset'''

    df = pd.read_csv('bank_cleaned.csv')
    df = df.sample(frac=1)
    # bank_data = df.sample(n = 10000)

    for i in range(4):
        if i == 7:
            bank_data = df.iloc[60000:79844]
            break
        bank_data = df.iloc[i*20000:i*20000+20000]

        y = bank_data['y'].values
        X = bank_data[['age','job','marital','education','default','balance',
                    'housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']].values


        fitness_obj = fitness.Fitness(nqubits,2,X,y,debug=True)

        training_features, training_labels, test_features, test_labels = fitness.Dataset(X,y)

        model = qsvm.QSVM(lambda parameters: fitness_obj.cc(pop, parameters)[0],training_features,training_labels)#fitness_obj(pop)

        y_pred = model.predict(test_features)

        cm = confusion_matrix(test_labels, y_pred)

        #cm_display = ConfusionMatrixDisplay(cm).plot()
        ConfusionMatrixDisplay.from_predictions(test_labels, y_pred)
        plt.show()
        recall = recall_score(test_labels, y_pred)
        acc  = accuracy_score(test_labels, y_pred)

        print(f'String = {pop},\n accuracy = {acc}, recall = {recall} for {i+1}th section of the data')
        
    return None

print(f'Performance testing finished after {time.time()-start} seconds')
featuremap_performance(iot_salidas.circ[0],nqubits)


gen = logbook.select("gen")
wc = logbook.chapters["wc"].select("min")
acc = logbook.chapters["acc"].select("max")


fig, ax1 = plt.subplots()
plt.figure(dpi = 100)
line1 = ax1.plot(gen, wc ,"b-", label="Min Weight Control")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Weight Control", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, acc, "r-", label="Max Accuracy")
ax2.set_ylabel("Accuracy", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="best")