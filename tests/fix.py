f = open('../data/points_5mln_4d_5c.txt', 'r')
fw = open('../data/points_5mln_4d_5c_fix.txt', 'w')
ok = False
for idx, line in enumerate(f):
    if not ok:
        ok = True
        continue
    point = list(line.strip().split(' '))
    fw.write(' '.join([str(idx)] + point) + '\n')
    print(idx)
f.close()
fw.close()