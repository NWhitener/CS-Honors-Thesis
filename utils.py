from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import random 
XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]





def evaluate2(model, preds, testing_target, testing_data): 
    metrics = []
    metrics.append(accuracy_score(testing_target,preds))
    metrics.append(f1_score(testing_target,preds))
    metrics.append(precision_score(testing_data,testing_target))
    metrics.append(recall_score(testing_data,testing_target))
    metrics.append(roc_auc_score(testing_data,testing_target))
    return metrics

def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.
    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 2  #Sex types
    n_features = 1000 # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))



def load_simulation_data(clientId = 1, balance = 0):
    if clientId == 1: 
        data = pd.read_csv("../client1.csv", index_col = 0)
        print("Loaded Dataset for Client: ", clientId)
        if balance == 1:
           print("HERE")
           X = data.iloc[:,:-1]  # the last column contains labels
           y = data.iloc[:, -1]
           x_train, y_train = X[:800], y[:800]
           x_test, y_test = X[200:], y[200:]
    elif clientId ==2: 
        data = pd.read_csv("../client2.csv", index_col = 0 )
        print("Loaded Dataset for Client: ", clientId)
        if balance == 2:
           X = data.iloc[:,:-1]  # the last column contains labels
           y = data.iloc[:, -1]
           x_train, y_train = X[:800], y[:800]
           x_test, y_test = X[200:], y[200:]
    elif clientId ==3: 
        data = pd.read_csv("../client3.csv", index_col = 0 )
        print("Loaded Dataset for Client: ", clientId)
        if balance == 3:
           X = data.iloc[:,:-1]  # the last column contains labels
           y = data.iloc[:, -1]
           x_train, y_train = X[:400], y[:400]
           x_test, y_test = X[100:], y[100:]
    else: 
        data = pd.read_csv("../client4.csv", index_col = 0 )
        print("Loaded Dataset for Client: ", clientId)    
    if balance == 0:
       X = data.iloc[:,:-1]  # the last column contains labels
       y = data.iloc[:, -1]
       x_train, y_train = X[:800], y[:800]
       x_test, y_test = X[200:], y[200:]
    return (x_train, y_train) , (x_test, y_test)

def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML.
    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # First 60000 samples consist of the train set
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    print(x_test.shape, y_test.shape) 
    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )


def load_kidney() -> Dataset:
    """Loads the MNIST dataset using OpenML.
    OpenML dataset link: https://www.openml.org/d/554
    """
    data = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/kidney.csv")
    labels = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/labels.csv", low_memory= False)

    data, labels = prepare_data(data, labels)

    data = data.iloc[:, 1:]
    # labels = encoding(labels)
    #    labels = pd.get_dummies(labels["SEX"])
    #   print(labels.value_counts())
    labels["SEX"].replace(["male", "female"], [0, 1], inplace=True)
    labels = labels["SEX"]
   # x_train, y_train = data[:234159], labels[:234159]
    #x_test, y_test = data[214159:], labels[214159:]
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=.2)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)

def load_kidney_single(clientId):
    if clientId == 1: 
        males = [3,4,5,7,8,9,10]
        females = [6,11,12]
        rMale = random.choice(males)
        rFemale = random.choice(females)
        datMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client1/dataMalePatient" + str(rMale)+ ".csv")
        datFemale =pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client1/dataFemalePatient" + str(rFemale)+ ".csv")
        labMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client1/labelsMalePatient" + str(rMale)+ ".csv")
        labFemale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client1/labelsFemalePatient" + str(rFemale)+ ".csv")
        frames = [labFemale,labMale]
        labels = pd.concat(frames)
        frames2 = [datFemale,datMale]
        data = pd.concat(frames2)
        print("Using Male Client " +str(rMale) +" and Female Client "+str(rFemale ))
    if clientId == 2:
        males = [15,16,17,20,21,22,23]
        females = [13,14,18]
        rMale = random.choice(males)
        rFemale = random.choice(females)
        datMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client2/dataMalePatient" + str(rMale)+ ".csv")
        datFemale =pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client2/dataFemalePatient" + str(rFemale)+ ".csv")
        labMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client2/labelsMalePatient" + str(rMale)+ ".csv")
        labFemale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client2/labelsFemalePatient" + str(rFemale)+ ".csv")
        frames = [labFemale,labMale]
        labels = pd.concat(frames)
        frames2 = [datFemale,datMale]
        data = pd.concat(frames2)
        print("Using Male Client " +str(rMale) +" and Female Client "+str(rFemale ))
    if clientId == 3:
        males = [24,25,26,28,29,30,31]
        females = [19,34]
        rMale = random.choice(males)
        rFemale = random.choice(females)
        datMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client3/dataMalePatient" + str(rMale)+ ".csv")
        datFemale =pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client3/dataFemalePatient" + str(rFemale)+ ".csv")
        labMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client3/labelsMalePatient" + str(rMale)+ ".csv")
        labFemale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/client3/labelsFemalePatient" + str(rFemale)+ ".csv")
        frames = [labFemale,labMale]
        labels = pd.concat(frames)
        frames2 = [datFemale,datMale]
        data = pd.concat(frames2)
        print("Using Male Client " +str(rMale) +" and Female Client "+str(rFemale ))
    if clientId == 4:
        datMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/server/dataMalePatient1.csv")
        datFemale =pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/server/dataFemalePatient2.csv")
        labMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/server/labelsMalePatient1.csv")
        labFemale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/server/labelsFemalePatient2.csv")
        frames = [labFemale,labMale]
        labels = pd.concat(frames)
        frames2 = [datFemale,datMale]
        data = pd.concat(frames2)
    if clientId == 5: 
        datMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/test/dataMalePatient32.csv")
        datFemale =pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/test/dataFemalePatient27.csv")
        labMale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/test/labelsMalePatient32.csv")
        labFemale = pd.read_csv("/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/test/labelsFemalePatient27.csv")
        frames = [labFemale,labMale]
        labels = pd.concat(frames)
        frames2 = [datFemale,datMale]
        data = pd.concat(frames2)
   # data, labels = prepare_data(data, labels)
    data = data.iloc[:, 2:]
    # labels = encoding(labels)
    #    labels = pd.get_dummies(labels["SEX"])
    #   print(labels.value_counts())
   
    labels["SEX"].replace(["male", "female"], [0, 1], inplace=True)
    labels = labels["SEX"]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1/7.0)

    return (x_train, y_train), (x_test, y_test)

def prepare_data(dataIn, labelsIn):
    """
    Help for the prepareData() function.

    Purpose: prepareData() takes a dataset and a labels set, and imposes the cell-id column as the defining feature. Do not use this
             function if the first column in the dataset is not the Cell-ID. Also, note that this function will not work after the
             readData function is used
    Return:  prepareData() will return the dataset and labels with the first column renamed to CELLID, for use of selectData,
             and other functions in the library
    Example: data, labels = prepareData(data,labelsSet)
    """
    dataIn.rename(columns={'Unnamed: 0': 'CELLID'}, inplace=True)
    labelsIn.rename(columns={'Unnamed: 0': 'CELLID'}, inplace=True)
    return dataIn, labelsIn


def renumber_did(label):
    """
  Help for the renumberDid() function.

  Purpose: renumberDid() takes the labels passed in through the sole required argument.
           This labels set need to be in the form of a Pandas DataFrame, this method will not
           automatically convert the label set.
  Return: renumberDid() will return the label set with the Patient Donor ID "DID" numbered from 0
           to the number of unique Patient Donor ID's
  Example: labels_renumbered = renumberDid(labels)
    """
    did_list_raw = label['DID'].value_counts().sort_index().index.values.tolist()
    did_list_renumbered = list(range(0, len(did_list_raw)))
    label_did_list = label['DID'].tolist()
    for num in range(0, len(label_did_list)):
        index = did_list_raw.index(label_did_list[num])
        label_did_list[num] = did_list_renumbered[index]
    label['DID'] = label_did_list
    return label


def selectData(data, col2select, ids):
    '''
    Help for the selectData() function. 

    Purpose: selectData() takes a data set, a column from the label set, and the match annotations to the dataset.
             With this information, selectData(), matches the data in the data file with the labels in the label 
             file based on the column. 
    Return:  selectData() will return data that matches the labels in based on the column selected 
           to the number of unique Patient Donor ID's
    Example: data = selectData(data, 'CELLID',labels['CELLID'].to_list())
    '''
    sample = data[data[col2select].isin(ids)]
    return sample

def splitPatients(data, labels):
    data, labels = prepare_data(data, labels)
    labels_men = labels.loc[labels['SEX'] == "male"]
    labels_women = labels.loc[labels['SEX'] == "female"]
    for i in labels_men["DID"].unique(): 
        labels_pat = labels_men[labels_men["DID"]==i]
        dataNew = selectData(data, "CELLID", labels_pat["CELLID"].to_list())
        dataPath = "/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/dataMalePatient" + str(i)+".csv"
        labelPath = "/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/labelsMalePatient" + str(i)+".csv"
        dataNew.to_csv(dataPath)
        labels_pat.to_csv(labelPath)
    for i in labels_women["DID"].unique(): 
        labels_pat = labels_women[labels_women["DID"]==i]
        dataNew = selectData(data, "CELLID", labels_pat["CELLID"].to_list())
        dataPath = "/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/dataFemalePatient" + str(i) +".csv"
        labelPath = "/Users/nathanwhitener/Desktop/DataForHP/SinglePatient/labelsFemalePatient" + str(i)+".csv"
        dataNew.to_csv(dataPath)
        labels_pat.to_csv(labelPath)




