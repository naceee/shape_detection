import subprocess
import os
from tqdm import tqdm

fnames = sorted(os.listdir('../point_clouds'))
num_points = 30
r_multiplier = 2

for fname in tqdm(list(filter(lambda x: x.startswith('4D'), fnames))):
    #if not fname.startswith('4D'):
    #    continue
    subprocess.run(['python', 'create_store_single_bns.py', fname, str(num_points), str(r_multiplier)])
    
