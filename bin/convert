#!/usr/bin/env python
from rpvanalysis import root2csv
import sys

if len(sys.argv) < 3:
    print(' specify input and output file')
    sys.exit(1)

r = root2csv.root2csv(sys.argv[1],sys.argv[2])
r.loop()
