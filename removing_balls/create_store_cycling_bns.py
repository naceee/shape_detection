import subprocess
import os
from tqdm import tqdm

fnames = sorted(os.listdir('./cycling_point_clouds'))
num_points = 0
r_multiplier = 2

for fname in fnames:
    #if not fname.startswith('4D'):
    #    continue
    subprocess.run(['python', 'create_store_single_cycling_bns.py', fname, str(num_points), str(r_multiplier)])
    
