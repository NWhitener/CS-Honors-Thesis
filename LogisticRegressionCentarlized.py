import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
import utils
from sklearn.model_selection import StratifiedKFold
import time

start = time.time()
data = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/kidney.csv")
labels = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/labels.csv",low_memory=False)
data, labels = utils.prepare_data(data, labels)
data = data.iloc[:, 1:]
labels = labels["SEX"]  

results = pd.DataFrame(columns=["Fold Number", "Accuracy"])

kfold = StratifiedKFold(n_splits=10, shuffle=True)


foldNumber = 1
for train, test in kfold.split(data, labels):
    accuracy = []
    # Train and Test Data
    training_data, testing_data = data.iloc[train], data.iloc[test],
    # Train and Test Labels
    training_label, testing_label = labels.iloc[train], labels.iloc[test]
    # Set up the model with its random state and 500 iterations
    # TODO: Tune Iterations
    logisticReg = LogisticRegression(max_iter=500)
    logisticReg.fit(training_data, training_label)
    predictions = logisticReg.predict(testing_data)
    score = logisticReg.score(testing_data, testing_label)
    accuracy.append(foldNumber)
    accuracy.append(score)
    foldNumber += 1
    cm = metrics.confusion_matrix(testing_label, predictions)
    results.loc[len(results.index)] = accuracy

end = time.time()
print("%s " %(results["Accuracy"].mean()))
print("Executed in %s" %(end - start) )
sb.lineplot(data=results, x="Fold Number", y="Accuracy")
plt.axhline(results["Accuracy"].mean(), color='r', ls="dashed")
# plt.axhline(0.748283084117, color='g', ls="dashed")
plt.show()
