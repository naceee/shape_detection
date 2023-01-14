import subprocess
import os
from tqdm import tqdm
from collections import defaultdict

#fnames = sorted(os.listdir('../point_clouds'))
num_points = 200
r_multiplier = 2

fpath = '../cyclists.csv'
with open(fpath, 'r') as f:
    data = f.readlines()
data = list(map(str.strip, data))
data = data[1:]
data = list(map(lambda x: x.split(','), data))
teams = defaultdict(list)
for item in data:
    _,_,_,mtn,_,tt,_,_,_,cob,spr,_,_,team = item
    teams[team].append([mtn,tt,cob,spr])

for team in teams:
    with open(f'./cycling_point_clouds/team_{team}.csv', 'w') as f:
        for rider in teams[team]:
            mtn, tt, cob, spr, = rider
            f.write(f'{mtn},{tt},{cob},{spr}\n')


# for fname in tqdm(list(filter(lambda x: x.startswith('4D'), fnames))):
#     #if not fname.startswith('4D'):
#     #    continue
#     subprocess.run(['python', 'create_store_single_bns.py', fname, str(num_points), str(r_multiplier)])
    
