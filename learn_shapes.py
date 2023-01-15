import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import pandas as pd
import warnings
from computing_persistent_homology import get_persistence_from_df
from persistent_images import create_persistence_image


def train_model(resolution):
    warnings.filterwarnings("ignore")

    data = pd.read_csv(f"persistence_images_data/data_n=100_resolution={resolution}.csv",
                       index_col="Unnamed: 0")
    X = data.loc[:, data.columns != 'result']
    y = data["result"]

    clf = RandomForestClassifier(bootstrap=True, criterion="gini", max_depth=16,
                                 n_estimators=4000)
    y_pred = cross_val_predict(clf, X, y, cv=5)

    classes = set(y)
    d = {c: i for i, c in enumerate(classes)}
    predictions = np.zeros((len(classes), len(classes)))
    for actual, predicted in zip(y, y_pred):
        predictions[d[actual], d[predicted]] += 1

    acc = np.trace(predictions) / np.sum(predictions)

    print(f"resolution: {resolution}, acc: {acc}")

    plt.rcParams["figure.figsize"] = (7, 7)
    plt.matshow(predictions)
    plt.xticks(list(range(len(d.keys()))), d.keys(), rotation=45)
    plt.yticks(list(range(len(d.keys()))), d.keys(), rotation=0)
    plt.savefig(f"images/acc_r={r}.pdf")
    plt.show()

    return acc


def predicting_cyclists(resolution=4):
    data = pd.read_csv(f"persistence_images_data/4D_data_n=100_resolution={resolution}.csv",
                       index_col="Unnamed: 0")
    data = data.dropna()
    X = data.loc[:, data.columns != 'result']
    y = data["result"]

    cyclists_df = pd.read_csv("cyclists.csv", index_col="Unnamed: 0")
    teams = set(cyclists_df["team"])

    clf = RandomForestClassifier(bootstrap=True, criterion="gini", max_depth=20,
                                 n_estimators=1000)
    clf.fit(X, y)

    print(f"{clf.classes_}")
    for team in teams:
        team_df = cyclists_df.loc[cyclists_df['team'] == team]
        team_df = team_df[["MOUNTAIN", "TIMETRAIL", "SPRINT", "COBBLE"]]

        team_persistence = get_persistence_from_df(team_df, max_dimensions=4)
        team_persistence.to_csv(f"cyclists_persistences/{team}.csv")

        persistence_image = create_persistence_image(f"cyclists_persistences/{team}.csv",
                                                     plot=False, resolution=resolution)

        print(team, clf.predict_proba(persistence_image.reshape(1, -1))[0])


if __name__ == "__main__":
    # predicting_cyclists(resolution=4)
    all_scores = []
    for r in range(2, 10):
        acc = train_model(r)
