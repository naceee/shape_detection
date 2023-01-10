from create_betti_sequences import *
import sys

if len(sys.argv) != 4:
    print("invalid arguments")
    exit()

fname = sys.argv[1]
num_points = int(sys.argv[2])
r_maxmin_distance_multiplier = float(sys.argv[3])

rpath = '../point_clouds/'+fname
V = V_from_file(rpath, num_points)
h = r_from_max_min_distance(V)
bnfname = fname.split('.')[0]+f'_n={num_points}_rm={r_maxmin_distance_multiplier}.txt'
bns = progressive_betti_numbers2(V, r_maxmin_distance_multiplier*h, 40+random.randint(0,10), bnfname)