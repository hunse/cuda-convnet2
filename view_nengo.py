import argparse
# import sys

import numpy as np
from run_nengo import error, view

parser = argparse.ArgumentParser(description="View Nengo run results.")
parser.add_argument('loadfile', help="File to load")
parser.add_argument('--view', action='store_const', const=True, default=False)
args = parser.parse_args()

# assert len(sys.argv) == 2
# loadfile = sys.argv[1]
objs = np.load(args.loadfile)

kwargs = {}
for n in ['dt', 'images', 'labels', 'data_mean', 'label_names', 't', 'y', 'z']:
    kwargs[n] = objs[n]

errors, y, z = error(*[kwargs[n] for n in ('dt', 'labels', 't', 'y', 'z')])
print "Error: %f" % errors.mean()
kwargs['y'] = y
kwargs['z'] = z

if args.view:
    view(**kwargs)
