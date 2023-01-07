import math
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os


def plane_equation(p1, p2, p3):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3

    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a, b, c, d


def plot_point_cloud(points, title="", figsize=(6, 6), num_points=500):
    # sorry this is very ugly
    warnings.filterwarnings("ignore")
    points = points[:num_points]

    fig = plt.figure(num=1, figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    # For each set of style and range settings, plot num_points random points in the box
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
        a = params["a"]
        b = params["b"]
        c = params["c"]
        area = 2*(a*b + b*c + a*c)
        probabilities = [2*a*b/area, 2*a*c/area, 2*b*c/area]

        for i in range(n):
            # Choose a random normal vector
            r = random.random()
            if r < probabilities[0]:
                x = random.uniform(0, a)
                y = random.uniform(0, b)
                z = random.choice([0, c]) + random.gauss(0, sigma)
            elif r < probabilities[0] + probabilities[1]:
                x = random.uniform(0, a)
                y = random.choice([0, b]) + random.gauss(0, sigma)
                z = random.uniform(0, c)
            else:
                x = random.choice([0, a]) + random.gauss(0, sigma)
                y = random.uniform(0, b)
                z = random.uniform(0, c)

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

    elif shape == "tetrahedron":
        e = params["edge_len"]
        p = [(e, 0, 0), (0, e, 0), (0, 0, e), (e, e, e)]
        planes = [plane_equation(p[0], p[1], p[2]),
                  plane_equation(p[0], p[1], p[3]),
                  plane_equation(p[0], p[2], p[3]),
                  plane_equation(p[1], p[2], p[3])]

        for i in range(n):
            side = random.randint(0, 3)

            x = random.uniform(0, e)
            y = random.uniform(0, e)
            z = (- planes[side][0] * x - planes[side][1] * y - planes[side][3]) / planes[side][2]

            while not 0 <= z <= e:
                x = random.uniform(0, e)
                y = random.uniform(0, e)
                z = (- planes[side][0] * x - planes[side][1] * y - planes[side][3])/planes[side][2]

            points[i, :] = [x, y, z]

    elif shape == "banana":
        phi_max = params["phi"]
        r_max = params["r"]
        R = params["R"]

        for i in range(n):
            phi = random.uniform(0, phi_max)
            r = np.power(np.sin(phi * np.pi / phi_max), 0.4) * r_max
            theta = random.uniform(0, 2 * np.pi)

            x = (R + r * math.cos(theta)) * math.cos(phi) + random.gauss(0, sigma)
            y = (R + r * math.cos(theta)) * math.sin(phi) + random.gauss(0, sigma)
            z = r * math.sin(theta) + random.gauss(0, sigma)

            points[i, :] = [x, y, z]

    elif shape == "filled_cube":
        edge_len = params["edge_len"]
        for i in range(n):
            x = random.uniform(-edge_len, edge_len)
            y = random.uniform(-edge_len, edge_len)
            z = random.uniform(-edge_len, edge_len)

            points[i, :] = [x, y, z]

    elif shape == "filled_sphere":
        r = params["r"]
        for i in range(n):
            x = random.uniform(-r, r)
            y = random.uniform(-r, r)
            z = random.uniform(-r, r)

            while x**2 + y**2 + z**2 > r:
                x = random.uniform(-r, r)
                y = random.uniform(-r, r)
                z = random.uniform(-r, r)

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


def create_rotated_point_clouds(shape_dicts, n, sigma, lower, upper):
    for shape, shape_dict in shape_dicts.items():

        for i in range(lower, upper):
            points = point_cloud(shape, n, sigma, shape_dict)
            points = rotate_around_axis(points, "x", random.uniform(0, 2 * math.pi))
            points = rotate_around_axis(points, "y", random.uniform(0, 2 * math.pi))
            points = rotate_around_axis(points, "z", random.uniform(0, 2 * math.pi))
            points = scale_points(points, random.uniform(1, 10))
            np.savetxt(f'point_clouds/{shape}_{i}.csv', points, delimiter=',')


def plot_shapes(shape=None, n=0):
    for filename in os.listdir("point_clouds"):
        matches = re.findall(f"_([0-9]*).csv", filename)

        if (len(matches) == 1 and int(matches[0]) <= n) and (shape is None or shape in filename):
            f = os.path.join("point_clouds", filename)
            points = np.loadtxt(f, delimiter=',')
            plot_point_cloud(points, filename)


def main():
    """
    for i in range(0, 100):
        a = random.uniform(1, 1.5)
        b = random.uniform(0.5, 1)
        c = random.uniform(0.2, 0.5)
        phi = random.uniform(math.pi / 4, math.pi)

        shape_dicts = {
            "sphere": {"r": 1},
            "torus": {"r": c, "R": a},
            "cube": {"edge_len": 1},
            "line": {"direction": (1, 1, 0.2), "length": 1},
            "cylinder": {"r": c, "height": a},
            "cuboid": {"a": a, "b": b, "c": c},
            "ellipsoid": {"a": a, "b": b, "c": c},
            "tetrahedron": {"edge_len": 1},
            "banana": {"phi": phi, "r": c, "R": a},
            "filled_sphere": {"r": 1},
            "filled_cube": {"edge_len": 1}
        }

        create_rotated_point_clouds(shape_dicts, 500, 0.01, i, i+1)
    """

    plot_shapes(shape="filled_cube", n=22)


if __name__ == "__main__":
    main()
