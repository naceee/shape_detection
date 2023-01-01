import copy
import math
import random
import re

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os


def plot_point_cloud(points, title=""):
    # sorry this is very ugly
    warnings.filterwarnings("ignore")

    fig = plt.figure(num=1, figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')

    # For each set of style and range settings, plot n random points in the box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect("equal")
    plt.title(title)
    plt.show()


def point_cloud(shape, n, sigma, params=None):
    points = np.zeros((n, 3))

    if shape == "sphere":
        r = params["r"]
        for i in range(n):
            # in this way you apparently get points spaced evenly on the surface
            x = np.random.normal(0, 1)
            y = np.random.normal(0, 1)
            z = np.random.normal(0, 1)

            l = math.sqrt(x ** 2 + y ** 2 + z ** 2)

            x = x * r / l + random.gauss(0, sigma)
            y = y * r / l + random.gauss(0, sigma)
            z = z * r / l + random.gauss(0, sigma)
            points[i, :] = [x, y, z]

    elif shape == "line":
        dir_v = params["direction"]
        length = params["length"]
        len_v = math.sqrt(dir_v[0] ** 2 + dir_v[1] ** 2 + dir_v[2] ** 2)
        dir_v = [v / len_v for v in dir_v]

        for i in range(n):
            k = random.uniform(-length / 2, length / 2)
            points[i, :] = [v * k + random.gauss(0, sigma) for v in dir_v]

    elif shape == "cube":
        edge_len = params["edge_len"]
        for i in range(n):
            p = [random.uniform(-edge_len, edge_len),
                 random.uniform(-edge_len, edge_len),
                 random.uniform(-edge_len, edge_len)]
            pick_xyz = random.randint(0, 2)
            pick_side = 2 * random.randint(0, edge_len) - edge_len + random.gauss(0, sigma)
            p[pick_xyz] = pick_side

            points[i, :] = p

    elif shape == "torus":
        r = params["r"]
        R = params["R"]
        for i in range(n):
            phi = random.uniform(0, 2 * math.pi)
            theta = random.uniform(0, 2 * math.pi)

            x = (R + r * math.cos(theta)) * math.cos(phi) + random.gauss(0, sigma)
            y = (R + r * math.cos(theta)) * math.sin(phi) + random.gauss(0, sigma)
            z = r * math.sin(theta) + random.gauss(0, sigma)
            points[i, :] = [x, y, z]

    elif shape == "cylinder":
        r = params["r"]
        h = params["height"]
        for i in range(n):
            surface_percentage_top_bottom = (2 * math.pi * r ** 2) / (2 * math.pi * r * (r + h))
            if random.random() < surface_percentage_top_bottom:
                x = random.uniform(-r, r)
                y = random.uniform(-r, r)
                while math.sqrt(x**2 + y**2) > r:
                    x = random.uniform(-r, r)
                    y = random.uniform(-r, r)

                z = random.choice([0, h]) + random.gauss(0, sigma)
            else:
                phi = random.uniform(0, 2 * math.pi)

                x = r * math.cos(phi) + random.gauss(0, sigma)
                y = r * math.sin(phi) + random.gauss(0, sigma)
                z = random.uniform(0, h) + random.gauss(0, sigma)
            points[i, :] = [x, y, z]

    elif shape == "cuboid":
        normals = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        a = params["a"]
        b = params["b"]
        c = params["c"]

        for i in range(n):
            # Choose a random normal vector
            normal = random.choice(normals)

            # Generate a random point on the corresponding face
            x = random.uniform(0, a) if normal[0] == 0 else (0 if normal[0] == 1 else a) + \
                random.gauss(0, sigma)
            y = random.uniform(0, b) if normal[1] == 0 else (0 if normal[1] == 1 else b) + \
                random.gauss(0, sigma)
            z = random.uniform(0, c) if normal[2] == 0 else (0 if normal[2] == 1 else c) + \
                random.gauss(0, sigma)

            points[i, :] = [x, y, z]

    elif shape == "ellipsoid":
        a = params["a"]
        b = params["b"]
        c = params["c"]
        for i in range(n):
            # Generate a random point in the bounding box
            x = random.uniform(-a, a)
            y = random.uniform(-b, b)
            z = random.uniform(-c, c)

            # Accept the point if it falls on the surface of the ellipsoid
            while not math.isclose(x**2 / a**2 + y**2 / b**2 + z**2 / c**2, 1, rel_tol=2*sigma):
                x = random.uniform(-a, a)
                y = random.uniform(-b, b)
                z = random.uniform(-c, c)
            points[i, :] = [x, y, z]

    else:
        raise Exception("invalid shape")

    return points


def rotate_around_axis(points, axis, phi):
    if axis == "z":
        rot_matrix = np.array([
            [math.cos(phi), -math.sin(phi), 0],
            [math.sin(phi), math.cos(phi), 0],
            [0, 0, 1]
        ])
    elif axis == "x":
        rot_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi), math.cos(phi)]
        ])
    elif axis == "y":
        rot_matrix = np.array([
            [math.cos(phi), 0, math.sin(phi)],
            [0, 1, 0],
            [-math.sin(phi), 0, math.cos(phi)]
        ])
    else:
        raise Exception("axis not 'x', 'y' or 'z'")

    return np.matmul(points, rot_matrix)


def translate_points(points, vector):
    return points + vector


def scale_points(points, k):
    return points * k


def create_point_clouds(shape_dicts, n, sigma):
    for shape, shape_dict in shape_dicts.items():
        points = point_cloud(shape, n, sigma, shape_dict)
        np.savetxt(f'point_clouds/{shape}.csv', points, delimiter=',')


def create_rotated_point_clouds(shape_dicts, n, sigma, k):
    for shape, shape_dict in shape_dicts.items():
        points = point_cloud(shape, n, sigma, shape_dict)

        for i in range(k):
            points_copy = copy.deepcopy(points)
            points_copy = rotate_around_axis(points_copy, "x", random.uniform(0, 2 * math.pi))
            points_copy = rotate_around_axis(points_copy, "y", random.uniform(0, 2 * math.pi))
            points_copy = rotate_around_axis(points_copy, "z", random.uniform(0, 2 * math.pi))
            points_copy = scale_points(points_copy, random.uniform(1, 10))
            np.savetxt(f'rotated_scaled_point_clouds/{shape}_{i}.csv', points_copy, delimiter=',')


def plot_all_shapes(shape=None):
    for filename in os.listdir("point_clouds"):
        if shape is None or shape in filename:
            f = os.path.join("point_clouds", filename)
            points = np.loadtxt(f, delimiter=',')
            plot_point_cloud(points, filename)


def plot_rotated_shapes(shape=None, n=0):
    for filename in os.listdir("rotated_scaled_point_clouds"):
        matches = re.findall(f"_([0-9]*).csv", filename)
        print(matches)

        if (len(matches) == 1 and int(matches[0]) <= n) and (shape is None or shape in filename):
            f = os.path.join("rotated_scaled_point_clouds", filename)
            points = np.loadtxt(f, delimiter=',')
            plot_point_cloud(points, filename)


def main():
    shape_dicts = {
        "sphere": {"r": 1},
        "torus": {"r": 0.2, "R": 1},
        "cube": {"edge_len": 1},
        "line": {"direction": (1, 1, 0.2), "length": 1},
        "cylinder": {"r": 1, "height": 1},
        "cuboid": {"a": 1, "b": 2, "c": 0.5},
        "ellipsoid": {"a": 1, "b": 2, "c": 0.5}
    }
    # create_point_clouds(shape_dicts, 500, 0.01)
    # create_rotated_point_clouds(shape_dicts, 500, 0.01, 100)

    # plot_all_shapes()
    plot_rotated_shapes()


if __name__ == "__main__":
    main()
