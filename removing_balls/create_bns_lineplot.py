from bns_classification import load_bns_from_file
import sys
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if len(sys.argv) != 2:
    print('invalid args')
    exit()
data = pd.DataFrame(columns=['step', 'betti number', 'dim'])
for i in range(100):
    bns, _ = load_bns_from_file(f'{sys.argv[1]}_{i}_n=200_rm=2.0.txt')
    #max_bn_dim = max(map(len, bns))
    #max_bn = max(map(max,bns))
    #bns = list(map(lambda x: x if len(x) == max_bn_dim else x + [0]*(max_bn_dim-len(x)), bns))
    #data.append(bns)
    for step, step_bns in enumerate(bns):
        for dim, bn in enumerate(step_bns):
            df = pd.DataFrame({'step':[step],'betti number':[bn],'dim':[dim]})
            data = pd.concat([data, df], ignore_index=True)
#print(bns)
# max_bns_len = 50
# ndata = []
# for i in range(len(data)):
#     bns = data[i]
#     nbns = []
#     for i in range(max_bns_len):
#         p = i / (max_bns_len)
#         pbns = p*(len(bns)-1)
#         il, ih = int(pbns), int(pbns)+1
#         w = pbns - int(pbns)
#         entry = [bns[il][d]*(1-w)+bns[ih][d]*w for d in range(3)]
#         nbns.append(entry)
#     ndata.append(nbns)
# data = ndata
# data = np

#seaborn.heatmap(list(zip(*bns)), square=True)
print(len(data))
seaborn.lineplot(data=data, x='step', y='betti number', hue='dim', palette=['red', 'green', 'blue', 'orange'])
plt.title(sys.argv[1])
plt.savefig(f'img/{sys.argv[1]}_lineplot.png')

# with open('./img/'+sys.argv[1][:-4]+'.pgm', 'w') as f:
#     f.write(f'P2\n{len(bns)} {max_bn_dim}\n{max_bn}\n')
#     for bnd in range(max_bn_dim):
#         for step in range(len(bns)):
#             f.write(f'{bns[step][bnd]} ')
#         f.write('\n')