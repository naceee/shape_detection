from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics


def train_model(resolution=4):
    data = pd.read_csv(f"persistence_images_data/data_n=100_resolution={resolution}.csv")
    X = data.loc[:, data.columns != 'result']
    y = data["result"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=30, n_estimators=1000)

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy on training data:", metrics.accuracy_score(y_train, clf.predict(X_train)))
    print("Accuracy on test data:", metrics.accuracy_score(y_test, y_pred))

    print("wrong predictions:")
    print("value     | predicted")
    print("---------------------")
    for (t, p) in zip(y_test, y_pred):
        if t != p:
            print(f"{t:9} | {p:9}")


if __name__ == "__main__":
    for r in range(2, 21):
        print("resolution:", r)
        train_model()
        print()
