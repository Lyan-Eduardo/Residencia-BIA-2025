'''
this script takes as input the output of evaluate_suffix_and_remaining_time.py
therefore, the latter needs to be executed first

Author: Niek Tax
'''

from __future__ import division
import csv

eventlog = "helpdesk.csv"

with open(
    'output_files/results/suffix_and_remaining_time_%s' % eventlog,
    'r',
    encoding='utf-8',
    newline=''
) as csvfile:
    r = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(r)  # header
    vals = dict()
    for row in r:
        l = vals.get(row[0], [])
        # row[1] e row[2] são strings das duas sequências que queremos comparar
        if len(row[1]) == 0 and len(row[2]) == 0:
            l.append(1)
        elif len(row[1]) == 0 and len(row[2]) > 0:
            l.append(0)
        elif len(row[1]) > 0 and len(row[2]) == 0:
            l.append(0)
        else:
            # 1 se o primeiro símbolo coincide, 0 caso contrário
            l.append(int(row[1][0] == row[2][0]))
        vals[row[0]] = l

l2 = []
for k, v in vals.items():
    l2.extend(v)
    res = sum(v) / len(v)
    print('{}: {}'.format(k, res))

print('total: {}'.format(sum(l2) / len(l2)))