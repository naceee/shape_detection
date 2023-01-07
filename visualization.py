import copy
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from persim import PersistenceImager


def plot_point_cloud(fig, position, points, n, title=""):
    ax = fig.add_subplot(position[0], position[1], position[2], projection='3d')

    # For each set of style and range settings, plot num_points random points in the box
    used_points = points[:n]
    not_used_points = points[n:]
    ax.scatter(used_points[0], used_points[1], used_points[2], color="blue")
    # ax.scatter(not_used_points[0], not_used_points[1], not_used_points[2], )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_PD(fig, position, df, title=""):
    ax = fig.add_subplot(position[0], position[1], position[2])

    dim2 = df[df["dimension"] == 2][["birth", "death"]]
    dim1 = df[df["dimension"] == 1][["birth", "death"]]
    dim0 = df[df["dimension"] == 0][["birth", "death"]]

    ax.scatter(dim2["birth"], dim2["death"], color="red", alpha=0.5, label="dim=2")
    ax.scatter(dim1["birth"], dim1["death"], color="blue", alpha=0.5, label="dim=1")
    ax.scatter(dim0["birth"], dim0["death"], color="green", alpha=0.5, label="dim=0")

    ax.set_xlabel('birth')
    ax.set_ylabel('death')
    plt.legend()
    ax.set_title(title)


def plot_PI(fig, position, df, title=""):
    ax = fig.add_subplot(position[0], position[1], position[2])
    df["persistence"] = df["death"] - df["birth"]

    dim2 = df[df["dimension"] == 2][["birth", "persistence"]]
    dim1 = df[df["dimension"] == 1][["birth", "persistence"]]
    dim0 = df[df["dimension"] == 0][["birth", "persistence"]]

    ax.scatter(dim2["birth"], dim2["persistence"], color="red", alpha=0.5, label="dim=2")
    ax.scatter(dim1["birth"], dim1["persistence"], color="blue", alpha=0.5, label="dim=1")
    ax.scatter(dim0["birth"], dim0["persistence"], color="green", alpha=0.5, label="dim=0")

    ax.set_xlabel('birth')
    ax.set_ylabel('persistence')
    plt.legend()
    ax.set_title(title)


def plot_normalized_PI(fig, position, df, title=""):
    ax = fig.add_subplot(position[0], position[1], position[2])
    df["persistence"] = df["death"] - df["birth"]

    df["birth"] = df["birth"] / df["birth"].max()
    df["persistence"] = df["persistence"] / df["persistence"].max()

    dim2 = df[df["dimension"] == 2][["birth", "persistence"]]
    dim1 = df[df["dimension"] == 1][["birth", "persistence"]]
    dim0 = df[df["dimension"] == 0][["birth", "persistence"]]

    ax.scatter(dim2["birth"], dim2["persistence"], color="red", alpha=0.5, label="dim=2")
    ax.scatter(dim1["birth"], dim1["persistence"], color="blue", alpha=0.5, label="dim=1")
    ax.scatter(dim0["birth"], dim0["persistence"], color="green", alpha=0.5, label="dim=0")

    ax.set_xlabel('normalized birth')
    ax.set_ylabel('normalized persistence')
    plt.legend()
    ax.set_title(title)


def create_persistence_image(fig, position, df, dim, resolution=5, title=""):
    ax = fig.add_subplot(position[0], position[1], position[2])

    df["pers_range"] = (df["death"] - df["birth"])
    df["birth"] = df["birth"] / df["birth"].max()
    df["pers_range"] = df["pers_range"] / df["pers_range"].max()

    H = df[df["dimension"] == dim][["birth", "pers_range"]].to_numpy()
    pimgr = PersistenceImager(pixel_size=(1 / resolution), birth_range=(0, 1), pers_range=(0, 1))
    pimgs = pimgr.transform(H, skew=False)

    pimgr.plot_image(pimgs, ax=ax)
    ax.set_title(title)


def visualize_pipeline(filename):
    warnings.filterwarnings("ignore")

    points = pd.read_csv(f"point_clouds/{filename}.csv", header=None)

    height = 3
    width = 3
    fig = plt.figure(figsize=(4*width, 4*height))

    plot_point_cloud(fig, (height, width, 1), points, 500, title="point cloud (500 points)")
    plot_point_cloud(fig, (height, width, 2), points, 200, title="point cloud (200 points)")

    persistence = pd.read_csv(f"persistence/{filename}_n=200.csv")
    persistence = persistence.replace([np.inf], np.nan).dropna()

    plot_PD(fig, (height, width, 4), copy.deepcopy(persistence), title="persistence diagram")
    plot_PI(fig, (height, width, 5), copy.deepcopy(persistence), title="persistence image")
    plot_normalized_PI(fig, (height, width, 6), copy.deepcopy(persistence),
                       title="normalized persistence image")

    create_persistence_image(fig, (height, width, 7), copy.deepcopy(persistence), 0, resolution=2,
                             title="")
    create_persistence_image(fig, (height, width, 8), copy.deepcopy(persistence), 1, resolution=2)
    create_persistence_image(fig, (height, width, 9), copy.deepcopy(persistence), 2, resolution=2)

    plt.show()


if __name__ == "__main__":
    visualize_pipeline("sphere_6")
