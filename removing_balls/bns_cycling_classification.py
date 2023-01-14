import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

data_path = '../betty_number_sequences/'

def load_bns_from_file(fname):
    with open(data_path+fname) as f:
        bns = f.readlines()
    bns = list(map(str.strip, bns))
    bns = list(map(lambda x: list(map(int, x.split(','))), bns))
    if 'team' in fname:
        tgt = 'team_'+fname[5:].split('_')[0]
    else:
        tgt = '4D_'+fname[3:].split('_')[0]
    return bns, tgt

if __name__ == '__main__':
    data = []
    targets = []



    max_bns_len = 0

    for fname in os.listdir(data_path):
        if not ((fname.startswith('4D') and 'n=30' in fname) or 'team' in fname):
            continue
        bns, tgt = load_bns_from_file(fname)
        #max_bn_dim = max(map(len, bns))
        max_bn_dim = 4
        bns = list(map(lambda x: x if len(x) == max_bn_dim else x + [0]*(max_bn_dim-len(x)), bns))
        max_bns_len = max(max_bns_len, len(bns))
        data.append(bns)
        targets.append(tgt)
    print('max_bns_length', max_bns_len)

    ndata = []
    for i in range(len(data)):
        bns = data[i]
        #print(bns ,i)
        #print(targets[i])
        nbns = []
        for i in range(max_bns_len):
            p = i / (max_bns_len)
            pbns = p*(len(bns)-1)
            il, ih = int(pbns), int(pbns)+1
            w = pbns - int(pbns)
            entry = [bns[il][d]*(1-w)+bns[ih][d]*w for d in range(4)]
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
    print(data[:4])

    #X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.3)
    print(targets)
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(data)):
        if 'team' in targets[i]:
            X_test.append(data[i])
            y_test.append(targets[i])
        else:
            X_train.append(data[i])
            y_train.append(targets[i])

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=30, n_estimators=1000)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    print(clf.classes_)
    print(y_pred)
    with open('team_classifications.csv', 'w') as f:
        f.write('team,'+','.join(clf.classes_)+'\n')
        for i in range(len(y_pred)):
            f.write(y_test[i]+','+','.join(map(str, y_pred[i]))+'\n')

    # print("Accuracy on training data:", metrics.accuracy_score(y_train, clf.predict(X_train)))
    # print("Accuracy on test data:", metrics.accuracy_score(y_test, y_pred))

    # print("wrong predictions:")
    # print("value     | predicted")
    # print("---------------------")
    # for (t, p) in zip(y_test, y_pred):
    #     if t != p:
    #         print(f"{t:9} | {p:9}")


