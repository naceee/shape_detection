import gudhi as gd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random


def V_from_file(path, n):
    with open(path, 'r') as f:
        data = f.readlines()
    points = list(map(lambda line: list(map(float, line.split(','))), data))
    if n != 0:
        return points[:n]
    else:
        return points



def r_from_max_min_distance(V, avg=5):
    dm = distance_matrix(V, V)
    h = sorted(list(map(lambda x: sorted(x)[1], dm)))[-1*avg:]
    h = sum(h) / avg
    #print(h)
    return h

def r_Rips_graph(V, steps=30):
    dm = distance_matrix(V, V)
    max_r = dm.max()
    steps = 30
    rs = []
    num_of_verts = []
    for step in range(steps):
        print(step)
        r = (max_r / steps) * step
        RC = gd.RipsComplex(distance_matrix=dm, max_edge_length=r)
        sxtree = RC.create_simplex_tree(max_dimension=3)
        rs.append(r)
        num_of_verts.append(sxtree.num_simplices())
    plt.plot(rs, num_of_verts)
    plt.show()


def progressive_betti_numbers(V, r):
    mid_point = list(map(lambda xs: sum(xs) / len(xs), zip(*V)))
    max_dist = max(map(lambda x: np.linalg.norm(np.array(mid_point)-np.array(x)), V))
    steps = 30
    bns = []
    for step in range(steps):
        print(step)
        fr = (max_dist / steps) * step
        FV = list(filter(lambda v: np.linalg.norm(np.array(v)-np.array(mid_point)) > fr, V))
        RC = gd.RipsComplex(points=FV, max_edge_length=r)
        sxtree = RC.create_simplex_tree(max_dimension=3)
        sxtree.compute_persistence()
        betti_nums = sxtree.betti_numbers()
        print(betti_nums)
        bns.append(betti_nums)
    print(bns)
    return bns


def progressive_betti_numbers2(V, r, steps=50, pcfname='0_pointcloud.txt'):
    mid_point = list(map(lambda xs: sum(xs) / len(xs), zip(*V)))
    max_dist = max(map(lambda x: np.linalg.norm(np.array(mid_point)-np.array(x)), V))
    #steps = 50
    bns = []
    RC = gd.RipsComplex(points=V, max_edge_length=r)
    ST = RC.create_simplex_tree(max_dimension=len(mid_point))
    ST.reset_filtration(0.0)
    open('../betty_number_sequences/'+pcfname, 'w').close()
    for step in range(steps):
        fr = (max_dist / steps) * step
        FV = list(filter(lambda v: np.linalg.norm(np.array(v)-np.array(mid_point)) < fr, V))
        inner_points = []
        for v in FV:
            inner_points.append(V.index(v))
        for ip in inner_points:
            for sx, v in ST.get_filtration():
                if ip in sx:
                    ST.assign_filtration(sx, 1.0)
        #print('updated filtration values')
        ST.prune_above_filtration(0.5)
        #print('pruned')
        ST.compute_persistence()
        #print('calculated persistence')
        betti_nums = ST.betti_numbers()
        #print(betti_nums)
        with open('../betty_number_sequences/'+pcfname, 'a') as f:
            f.write(','.join(map(str,betti_nums)) + '\n')
        #print('calculated betti numbers')
        #print(step, betti_nums)
        bns.append(betti_nums)
    #print(bns)
    return bns


def write_bns_to_file(fname, bns):
    with open('./betty_number_sequences/'+fname, 'w') as f:
        for bn in bns:
            f.write(','.join(map(str,bn)) + '\n')


def create_betty_seq_data(num_points=200, r_maxmin_distance_multiplier=2.5):
    fnames = sorted(os.listdir('./point_clouds'))
    for fname in tqdm(fnames[42:]):
        rpath = './point_clouds/'+fname
        V = V_from_file(rpath, num_points)
        h = r_from_max_min_distance(V)
        bns = progressive_betti_numbers2(V, r_maxmin_distance_multiplier*h, 40+random.randint(0,10))
        write_bns_to_file(
            fname.split('.')[0]+f'_n={num_points}_rm={r_maxmin_distance_multiplier}.txt',
            bns)



if __name__ == '__main__':
    #pass
    path = '../point_clouds/4D_cube_0.csv'
    V = V_from_file(path, 200)
    #r_Rips_graph(V)
    h = r_from_max_min_distance(V)
    progressive_betti_numbers2(V,2*h)
    #create_betty_seq_data()