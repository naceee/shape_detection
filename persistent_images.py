import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from persim import PersistenceImager
import warnings


def get_data_from_persistence_file(filename):
    persistence = pd.read_csv(filename)
    persistence = persistence.replace([np.inf], np.nan).dropna()

    persistence["pers_range"] = (persistence["death"] - persistence["birth"])
    persistence["birth"] = persistence["birth"] / persistence["birth"].max()
    persistence["death"] = persistence["death"] / persistence["death"].max()
    persistence["pers_range"] = persistence["pers_range"] / persistence["pers_range"].max()

    H2 = persistence[persistence["dimension"] == 2][["birth", "pers_range"]].to_numpy()
    H1 = persistence[persistence["dimension"] == 1][["birth", "pers_range"]].to_numpy()
    H0 = persistence[persistence["dimension"] == 0][["birth", "pers_range"]].to_numpy()

    return [H0, H1, H2], persistence


def create_persistence_image(filename, plot=False, resolution=5):
    warnings.filterwarnings("ignore")

    persistence_array, persistence = get_data_from_persistence_file(filename)
    pimgr = PersistenceImager(pixel_size=(1 / resolution), birth_range=(0, 1), pers_range=(0, 1))

    if plot:
        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        axs[0].set_title("PD (birth - death)")
        colors = ["blue", "green", "red"]
        for dim in [0, 1, 2]:
            axs[0].scatter(persistence[persistence["dimension"] == dim]["birth"],
                           persistence[persistence["dimension"] == dim]["death"],
                           color=colors[dim], alpha=0.5)
        axs[0].axis("equal")

        axs[1].set_title("PD (birth - persistence)")
        for dim in [0, 1, 2]:
            axs[1].scatter(persistence[persistence["dimension"] == dim]["birth"],
                           persistence[persistence["dimension"] == dim]["pers_range"],
                           color=colors[dim], alpha=0.5)
        axs[1].axis("equal")

    result = []
    for i, pers_dgm in enumerate(persistence_array):
        pimgs = pimgr.transform(pers_dgm, skew=False)
        result.append(pimgs)

        if plot:
            axs[i+2].set_title(f"Persistence Image for H_{i}")
            pimgr.plot_image(pimgs, ax=axs[i+2])
    if plot:
        plt.tight_layout()
        plt.show()

    return np.array(result).flatten()


def create_training_set(n=40, resolution=5):
    training_set = []
    for shape in ["cube", "cuboid", "cylinder", "ellipsoid", "line", "sphere", "torus"]:
        for i in range(n):
            x = create_persistence_image(f"persistence/{shape}_{i}_n=200.csv",
                                         plot=False, resolution=resolution)
            training_set.append(list(x) + [shape])
    training_df = pd.DataFrame(training_set, columns=(list(range(3*resolution**2)) + ["result"]))
    training_df.to_csv(f"persistence_images_data/data_n={n}_resolution={resolution}.csv")


def main():
    for r in range(2, 21):
        create_training_set(n=100, resolution=r)


if __name__ == "__main__":
    main()
