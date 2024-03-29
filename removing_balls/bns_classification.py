import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

data_path = '../betty_number_sequences/'

def load_bns_from_file(fname):
    with open(data_path+fname) as f:
        bns = f.readlines()
    bns = list(map(str.strip, bns))
    bns = list(map(lambda x: list(map(int, x.split(','))), bns))
    if 'filled' in fname:
        tgt = 'filled_'+fname[7:].split('_')[0]
    else:
        tgt = fname.split('_')[0]
    return bns, tgt

if __name__ == '__main__':
    data = []
    targets = []



    max_bns_len = 0

    for fname in os.listdir(data_path):
        if fname.startswith('4D') or 'team' in fname:
            continue
        bns, tgt = load_bns_from_file(fname)
        max_bn_dim = max(map(len, bns))
        bns = list(map(lambda x: x if len(x) == max_bn_dim else x + [0]*(max_bn_dim-len(x)), bns))
        max_bns_len = max(max_bns_len, len(bns))
        data.append(bns)
        targets.append(tgt)

    ndata = []
    for i in range(len(data)):
        bns = data[i]
        nbns = []
        for i in range(max_bns_len):
            p = i / (max_bns_len)
            pbns = p*(len(bns)-1)
            il, ih = int(pbns), int(pbns)+1
            w = pbns - int(pbns)
            entry = [bns[il][d]*(1-w)+bns[ih][d]*w for d in range(3)]
            nbns.append(entry)
        ndata.append(nbns)
    data = ndata
    shapes = set()
    for bns in data:
        shapes.add(np.array(bns).shape)
    print(shapes)

    data = list(map(lambda bns: np.concatenate(np.array(bns, dtype=float).T), data))
    data = np.array(data)
    print('the data:')
    print(data[:5])
    print(targets[:5])

    #X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.3)

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=30, n_estimators=1000)

    # Train the model using the training sets
    #clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)

    # print("Accuracy on training data:", metrics.accuracy_score(y_train, clf.predict(X_train)))
    # print("Accuracy on test data:", metrics.accuracy_score(y_test, y_pred))

    # print("wrong predictions:")
    # print("value     | predicted")
    # print("---------------------")
    # for (t, p) in zip(y_test, y_pred):
    #     if t != p:
    #         print(f"{t:9} | {p:9}")

    y_pred = cross_val_predict(clf, data, targets, cv=5)

    classes = set(targets)
    d = {c: i for i, c in enumerate(classes)}
    predictions = np.zeros((len(classes), len(classes)))
    for actual, predicted in zip(targets, y_pred):
        predictions[d[actual], d[predicted]] += 1

    acc = np.trace(predictions) / np.sum(predictions)

    #print(f"resolution: {resolution}, acc: {acc}")

    plt.rcParams["figure.figsize"] = (7, 7)
    plt.matshow(predictions)
    plt.xticks(list(range(len(d.keys()))), d.keys(), rotation=45)
    plt.yticks(list(range(len(d.keys()))), d.keys(), rotation=0)
    plt.savefig("classification.png")
    plt.show()


