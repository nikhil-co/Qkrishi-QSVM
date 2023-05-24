'''
Python Script to compare Mutation Probability with Accuracy
'''


from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import qsvm
import fitness
import gsvm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, accuracy_score

def ordenar_salidas_pareto(dataframe):
    dataframe.columns=['ind','circ','gates','accuracy']
    dataframe.sort_values(['gates','accuracy'], ascending=[False,False],inplace=True)
    dataframe.reset_index(inplace=True)
    dataframe.pop('index')
    return dataframe

def featuremap_performance(pop:str,nqubits:int) -> None:
    '''Returns the performance of a feature map on all of the dataset'''
    df_1 = df.sample(frac=1)
    for i in range(8):
        if i == 7:
            bank_data = df_1.iloc[70000:79844]
            break
        bank_data = df_1.iloc[i*10000:i*10000+10000]

        y = bank_data['y'].values
        X = bank_data[['age','job','marital','education','default','balance',
                    'housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']].values

        fitness_obj = fitness.Fitness(nqubits,16,X,y,debug=True)

        training_features, training_labels, test_features, test_labels = fitness.Dataset(X,y)

        model = qsvm.QSVM(lambda parameters: fitness_obj.cc(pop, parameters)[0],training_features,training_labels)

        y_pred = model.predict(test_features)

        cm = confusion_matrix(test_labels, y_pred)

        ConfusionMatrixDisplay.from_predictions(test_labels, y_pred)
        plt.show()
        recall = recall_score(test_labels, y_pred)
        acc  = accuracy_score(test_labels, y_pred)
        line = f'Accuracy = {acc}, Recall = {recall} for {i+1}th section of the data'
        print(line)
        with open('bank_out.csv','a') as f:
            f.write(line)
            f.write('\n')
    return None

n_samples = 2000
nqubits = 6
depth = 6
mu = 50
lambda_ = 25    
ngen = 10

divisions = 11

output="bank_testdata.csv"
df = pd.read_csv('bank_cleaned.csv')

bank_data = df.sample(n=n_samples,random_state=13)

y = bank_data['y'].values
X = bank_data[['age','job','marital','education','default','balance','housing','loan','contact',
                'day','month','duration','campaign','pdays','previous','poutcome']].values

acc_list = []
mut_prob = np.linspace(0,1,divisions)
start = time.time()
for i in mut_prob:
    
    pop, pareto, logbook = gsvm.gsvm(nqubits=nqubits, depth=depth, nparameters=16,
                                        X=X, y=y, weights=[-1.0,1.0],
                                        mu=mu,lambda_=lambda_,ngen=ngen,mutpb=i,cxpb=1-i,debug=False)

    with open(output, "w") as f:
        for ide, ind in enumerate(pareto):
            genes=''.join(str(i) for i in list(ind))
            gates, acc = ind.fitness.values
            line = f'{ide},"{genes}",{gates},{acc}'
            f.write(line)
            f.write('\n')

    iot_result = pd.read_csv('bank_testdata.csv',header=None)

    iot_salidas = ordenar_salidas_pareto(iot_result)

    # gen = logbook.select("gen")
    # wc = logbook.chapters["wc"].select("media")
    # acc = logbook.chapters["acc"].select("media")


    # fig, ax1 = plt.subplots()
    # plt.figure(dpi = 100)
    # line1 = ax1.plot(gen, wc ,"b-", label="Avg Weight Control")
    # ax1.set_xlabel("Generation")
    # ax1.set_ylabel("Weight Control", color="b")
    # for tl in ax1.get_yticklabels():
    #     tl.set_color("b")

    # ax2 = ax1.twinx()
    # line2 = ax2.plot(gen, acc, "r-", label="Avg Accuracy")
    # ax2.set_ylabel("Accuracy", color="r")
    # for tl in ax2.get_yticklabels():
    #     tl.set_color("r")

    # lns = line1 + line2
    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc="best")

    #fig.savefig(f'evol_genplots/{nqubits},{depth},{mu},{lambda_},{ngen},{n_samples},{i}.png')
    acc_list.append(iot_salidas['accuracy'][0])

print(f'Simulation finished after {time.time()-start} seconds\n\n')

plt.figure(figsize=(14,7))
plt.title('Mutation Probability vs Accuracy')
plt.plot(mut_prob,acc_list,marker='P',ls='--')
plt.xlim([-.1,1.1])
plt.ylim([0,1])
plt.xticks(np.linspace(0,1,11))
plt.grid()