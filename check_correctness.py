f = open('data/results_5mln_4d_5c.txt', 'r')
g = open('out.txt', 'r')

for _ in range(5):
    f_r = f.readline()
    g_r = g.readline()

for i in range(5_000_000):
    f_line = f.readline().strip()
    g_line = g.readline().strip()
    if f_line != g_line:
        print(f"Mismatch at line {i+6}: File f: {f_line} File g: {g_line}")

f.close()
g.close()