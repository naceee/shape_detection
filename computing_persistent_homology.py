import numpy as np
import pandas as pd
import gudhi as gd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re
import time


def get_persistence_dataframe(f_name, num_points=100, max_dimensions=3):
    df = pd.read_csv(f_name)
    df = df.iloc[:num_points]
    D = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)

    skeleton = gd.RipsComplex(
        distance_matrix=D.values,
        max_edge_length=100
    )

    rips_simplex_tree = skeleton.create_simplex_tree(max_dimension=max_dimensions)
    bar_codes_rips = rips_simplex_tree.persistence()
    bar_codes_rips = [[x[0], x[1][0], x[1][1]] for x in bar_codes_rips]

    persistence = pd.DataFrame(bar_codes_rips, columns=["dimension", "birth", "death"])
    return persistence


def plot_persistence_diagram(persistence_df, shape=None, save_filename=None):
    warnings.filterwarnings("ignore")

    m = persistence_df.loc[persistence_df['death'] != np.inf, 'death'].max()
    persistence_df['death'].replace(np.inf, m, inplace=True)

    sns.scatterplot(persistence_df, x="birth", y="death", hue="dimension",
                    palette=sns.color_palette("hls", 3))
    if shape is not None:
        plt.title(f"persistence diagram of {shape}")
    plt.xlim((-0.1 * m, m * 1.1))
    plt.ylim((-0.1 * m, m * 1.1))
    plt.legend(loc='lower right')
    if save_filename is not None:
        plt.savefig(f"persistence_diagrams/{save_filename}.pdf")

    plt.show()


def save_persistence_dataframes(num_points=200):
    persistence_list = os.listdir("persistence")
    print(persistence_list)

    for filename in os.listdir("point_clouds"):
        persistence_filename = f"{filename[:-4]}_n={num_points}.csv"
        if persistence_filename not in persistence_list:
            print("computing persistent homology for", persistence_filename)
            start = time.time()
            persistence_list.append(filename)
            persistence_df = get_persistence_dataframe(f"point_clouds/{filename}",
                                                       num_points=num_points, max_dimensions=3)
            persistence_df.to_csv(f"persistence/{persistence_filename}")
            end = time.time()
            print(f"elapsed time = {end - start} s")
        else:
            print("persistent homology for", persistence_filename, "is already computed")


def plot_persistence_diagrams(shape=None, n=0, save=False):
    for filename in os.listdir("persistence"):
        matches = re.findall(f"([a-z]*)_([0-9]*)_n=[0-9]*.csv", filename)

        if (len(matches) == 1 and int(matches[0][1]) <= n) and (shape is None or shape in filename):
            persistence_df = pd.read_csv(os.path.join("persistence", filename))
            if save:
                plot_persistence_diagram(persistence_df, matches[0][0], save_filename=filename[:-4])
            else:
                plot_persistence_diagram(persistence_df, matches[0][0])


def main():
    # save_persistence_dataframes()
    plot_persistence_diagrams(n=20, save=True)


if __name__ == "__main__":
    main()
