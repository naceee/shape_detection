import subprocess
import os
from tqdm import tqdm

fnames = sorted(os.listdir('../point_clouds'))
num_points = 200
r_multiplier = 2.5

def fnamefilter(fname):
    return 'tetra' in fname or 'filled' in fname or 'banana' in fname

for fname in tqdm(list(filter(fnamefilter, fnames))):
    #if not fname.startswith('4D'):
    #    continue
    subprocess.run(['python', 'create_store_single_bns.py', fname, str(num_points), str(r_multiplier)])
    
