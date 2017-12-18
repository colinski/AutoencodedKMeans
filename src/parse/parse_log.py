#!/usr/bin/env python
import sys
def parse_file(path):
    lines = open(path).readlines()
    rs, mi, v, p = 0, 0, 0, 0

    for line in lines:
        tokens = line.strip().split('\t')
        if len(tokens) > 3:
            rs = max(rs, float(tokens[1]))
            mi = max(mi, float(tokens[2]))
            v = max(v, float(tokens[3]))
            p = max(p, float(tokens[4]))
        else:
            continue

    print 'rs: %s mi: %s v_score: %s purity: %s' % (rs, mi, v, p)
    return [rs, mi, v, p]

if __name__ == '__main__':
    parse_file(sys.argv[1])
