import math
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings


def plot_point_cloud(points):
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
            phi = random.uniform(0, 2 * math.pi)

            x = r * math.cos(phi) + random.gauss(0, sigma)
            y = r * math.sin(phi) + random.gauss(0, sigma)
            z = random.uniform(0, h) + random.gauss(0, sigma)
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


def create_point_clouds(shape_dicts, n, sigma):
    for shape, shape_dict in shape_dicts.items():
        points = point_cloud(shape, n, sigma, shape_dict)
        np.savetxt(f'point_clouds/{shape}.csv', points, delimiter=',')


def create_rotated_point_clouds(shape_dicts, n, sigma, k):
    for shape, shape_dict in shape_dicts.items():
        points = point_cloud(shape, n, sigma, shape_dict)

        for i in range(k):
            points = rotate_around_axis(points, "x", random.uniform(0, 2 * math.pi))
            points = rotate_around_axis(points, "y", random.uniform(0, 2 * math.pi))
            points = rotate_around_axis(points, "z", random.uniform(0, 2 * math.pi))

            np.savetxt(f'rotated_point_clouds/{shape}_{i}.csv', points, delimiter=',')


def plot_shapes(shape, rotated=True, indices=None):
    if rotated:
        for idx in indices:
            points = np.loadtxt(f'rotated_point_clouds/{shape}_{idx}.csv', delimiter=',')
            plot_point_cloud(points)
    else:
        points = np.loadtxt(f'point_clouds/{shape}.csv', delimiter=',')
        plot_point_cloud(points)


def main():
    shape_dicts = {
        "sphere": {"r": 1},
        "torus": {"r": 0.2, "R": 1},
        "cube": {"edge_len": 1},
        "line": {"direction": (1, 1, 0.2), "length": 1},
        "cylinder": {"r": 1, "height": 2}
    }
    # create_point_clouds(shape_dicts, 500, 0.01)
    # create_rotated_point_clouds(shape_dicts, 500, 0.01, 10)

    plot_shapes("cylinder", True, list(range(10)))


if __name__ == "__main__":
    main()
