import subprocess
import os
from tqdm import tqdm

fnames = sorted(os.listdir('../point_clouds'))
num_points = 200
r_multiplier = 2.5

for fname in tqdm(fnames):
    subprocess.run(['python', 'create_store_single_bns.py', fname, str(num_points), str(r_multiplier)])
    
