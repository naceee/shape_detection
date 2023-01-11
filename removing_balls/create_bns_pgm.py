from bns_classification import load_bns_from_file
import sys

if len(sys.argv) != 2:
    print('invalid args')
    exit()

bns, _ = load_bns_from_file(sys.argv[1])
max_bn_dim = max(map(len, bns))
max_bn = max(map(max,bns))
bns = list(map(lambda x: x if len(x) == max_bn_dim else x + [0]*(max_bn_dim-len(x)), bns))
#print(bns)

with open('./img/'+sys.argv[1][:-4]+'.pgm', 'w') as f:
    f.write(f'P2\n{len(bns)} {max_bn_dim}\n{max_bn}\n')
    for bnd in range(max_bn_dim):
        for step in range(len(bns)):
            f.write(f'{bns[step][bnd]} ')
        f.write('\n')