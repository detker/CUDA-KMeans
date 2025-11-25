import numpy as np
from sklearn.datasets import make_moons, make_circles
import struct
import subprocess

SEED = 42
np.random.seed(SEED)

def write_binary(filename, data, k):
    """
    filename : output path
    data     : numpy array (N x d), dtype float64
    k        : number of clusters
    """

    data = data.astype(np.float64, copy=False)
    N, d = data.shape

    with open(filename, "wb") as f:
        # Write header: N, d, k as 4-byte int32
        f.write(struct.pack("<iii", N, d, k))

        # Write point coordinates as float64
        f.write(data.tobytes(order="C"))

    print(f"[saved] {filename} | shape=({N}, {d}), k={k}")

def write_text(filename, data, k):
    """
    filename : output path
    data     : numpy array (N x d), dtype float64
    k        : number of clusters
    """

    data = data.astype(np.float64, copy=False)
    N, d = data.shape

    with open(filename, "w") as f:
        # Write header: N, d, k
        # f.write(f"{N} {d} {k}\n")

        # Write point coordinates
        for i, point in enumerate(data):
            f.write(str(i+1) + " " + " ".join(map(str, point)) + "\n")

    print(f"[saved] {filename} | shape=({N}, {d}), k={k}")


# -------------------------------------------------------------------
# Dataset generators
# -------------------------------------------------------------------
def gen_uniform(N, d, k):
    return np.random.random((N, d))

def gen_gaussian_clusters(N, d, k, spread=0.1):
    centers = np.random.uniform(-10, 10, (k, d))
    pts_per_cluster = N // k
    data = []
    for i in range(k):
        pts = centers[i] + np.random.randn(pts_per_cluster, d) * spread
        data.append(pts)
    data = np.vstack(data)
    if data.shape[0] < N:
        # pad if division remainder
        extra = N - data.shape[0]
        data = np.vstack([data, centers[0] + np.random.randn(extra, d) * spread])
    return data

def gen_overlapping_clusters(N, d, k):
    centers = np.random.uniform(-3, 3, (k, d))
    pts_per_cluster = N // k
    data = []
    for i in range(k):
        pts = centers[i] + np.random.randn(pts_per_cluster, d) * 1.0
        data.append(pts)
    data = np.vstack(data)
    if data.shape[0] < N:
        extra = N - data.shape[0]
        data = np.vstack([data, centers[0] + np.random.randn(extra, d)])
    return data

def gen_skewed_density(N, d, k):
    centers = np.random.uniform(-20, 20, (k, d))
    big = int(N * 0.85)
    small = max((N - big) // (k - 1), 1)

    data = [centers[0] + 0.05 * np.random.randn(big, d)]

    for i in range(1, k):
        pts = centers[i] + 0.3 * np.random.randn(small, d)
        data.append(pts)

    data = np.vstack(data)
    if data.shape[0] < N:
        extra = N - data.shape[0]
        data = np.vstack([data, centers[0] + np.random.randn(extra, d)])
    return data

def gen_highdim(N, d, k):
    return np.random.randn(N, d)

def read_output_mine(filename):
    clusters = []
    assignments = []
    f = open(filename, 'r')
    point = None
    for line in f:
        point = list(map(float, line.strip().split(' ')))
        if len(point) == 1:
            # its a assigmnent idx
            assignments.append(int(point[0]))
        else: clusters.append(point)
    f.close()

    return clusters, assignments

def read_output_notmine(filename):
    clusters = []
    assignments = []

    f_clusters = open(filename+'.cluster_centres', 'r')
    for line in f_clusters:
        point = list(map(float, line.strip().split(' ')))[1:]
        clusters.append(point)
    f_clusters.close()

    f_assignments = open(filename+'.membership', 'r')
    for line in f_assignments:
        idx = int(line.strip().split(' ')[1])
        assignments.append(idx)
    f_assignments.close()

    return clusters, assignments

def launch_mine(data_format, compute_method, input_file, output_file):
    cmd = ['../build/KMeans',
            data_format,
            compute_method,
            input_file,
            output_file]

    subprocess.run(cmd, check=True)

def launch_notmine(data_format, input_file, n_clusters):
    cmd = ['../parallel-kmeans/seq_main',
            '-b' if data_format == 'bin' else '',
            '-n', str(n_clusters),
            '-i', input_file]

    subprocess.run(cmd, check=True)

def compare_results(clusters_mine, assignments_mine, clusters_notmine, assignments_notmine):
    clusters_mine_sorted = sorted(clusters_mine, key=lambda x: x[0])
    clusters_notmine_sorted = sorted(clusters_notmine, key=lambda x: x[0])

    assert len(clusters_mine_sorted) == len(clusters_notmine_sorted), "Number of clusters differ"

    for c1, c2 in zip(clusters_mine_sorted, clusters_notmine_sorted):
        assert np.allclose(c1, c2, atol=1e-6), f"Cluster centers differ: {c1} vs {c2}"
    
    assert len(assignments_mine) == len(assignments_notmine), "Number of assignments differ"

    for a1, a2 in zip(assignments_mine, assignments_notmine):
        assert a1 == a2, f"Assignments differ: {a1} vs {a2}"

# if __main__ == "__name__":
if __name__ == "__main__":

    ### 1st. test - sample data
    # launch_mine('bin', 'gpu1', '../data/points_5mln_4d_5c.dat', 'output_mine_gpu.txt')
    # # print('Mine kmeans finished')
    # print('KMeans GPU finished')
    # # launch_notmine('txt', '../data/points_5mln_4d_5c_fix.txt', 5)
    # launch_mine('bin', 'cpu', '../data/points_5mln_4d_5c.dat', 'output_mine_cpu.txt')
    # # print('Not mine kmeans finished')
    # print('KMeans CPU finished')

    # clusters_mine, assignments_mine = read_output_mine('output_mine_gpu.txt')
    # clusters_mine_cpu, assignments_mine_cpu = read_output_mine('output_mine_cpu.txt')
    # # clusters_notmine, assignments_notmine = read_output_notmine('../data/points_5mln_4d_5c_fix.txt')
    # print('Klastry:')
    # print(clusters_mine)
    # print()
    # print(clusters_mine_cpu)
    # print()
    # compare_results(clusters_mine, assignments_mine, clusters_mine_cpu, assignments_mine_cpu)
    # print('Alles klar!')

    configs = [
        ("uniform_big.txt",            gen_uniform,            1_200_000, 3, 10),
        # ("gaussian_big.txt",           gen_gaussian_clusters,  1_000_000, 3, 8),
        ("overlap_big.txt",            gen_overlapping_clusters, 900_000, 3, 8),
        # ("skewed_big.txt",             gen_skewed_density,     1_500_000, 3, 12),
        # ("highdim_big.txt",            gen_highdim,            500_000, 3, 20),
    ]
    OUT = '../data/'
    for filename, generator, N, d, k in configs:
        print(f"\nGenerating {filename} ...")
        data = generator(N, d, k) if generator != gen_highdim or generator != gen_uniform else generator(N, d)
        # write_text(OUT / filename, data, k)
        write_binary(OUT + (filename.replace(".txt", ".dat")), data, k)
        print(f"Dataset {filename} generated.")

        ## testing
        launch_mine('bin', 'gpu1', str(OUT + (filename.replace(".txt", ".dat"))), 'output_mine_gpu.txt')
        print('KMeans GPU finished')
        launch_mine('bin', 'cpu', str(OUT + (filename.replace(".txt", ".dat"))), 'output_mine_cpu.txt')
        print('KMeans CPU finished')
        clusters_mine, assignments_mine = read_output_mine('output_mine_gpu.txt')
        clusters_mine_cpu, assignments_mine_cpu = read_output_mine('output_mine_cpu.txt')
        compare_results(clusters_mine, assignments_mine, clusters_mine_cpu, assignments_mine_cpu)
        print('Alles klar with test: {}'.format(filename))
