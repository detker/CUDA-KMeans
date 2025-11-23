import numpy as np
import struct
from pathlib import Path

SEED = 42
np.random.seed(SEED)

OUT = Path("big_kmeans_datasets")
OUT.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Utility: write binary file in required format
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Main: generate BIG datasets
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example large dataset parameters
    # configs = [
    #     ("uniform_big.dat",            gen_uniform,            1_200_000, 3, 10),
    #     ("gaussian_big.dat",           gen_gaussian_clusters,  1_000_000, 3, 8),
    #     ("overlap_big.dat",            gen_overlapping_clusters, 900_000, 3, 8),
    #     ("skewed_big.dat",             gen_skewed_density,     1_500_000, 3, 12),
    #     ("highdim_big.dat",            gen_highdim,            500_000, 3, 20),
    # ]

    configs = [
        ("uniform_big.txt",            gen_uniform,            1_200_000, 3, 10),
        ("gaussian_big.txt",           gen_gaussian_clusters,  1_000_000, 3, 8),
        ("overlap_big.txt",            gen_overlapping_clusters, 900_000, 3, 8),
        ("skewed_big.txt",             gen_skewed_density,     1_500_000, 3, 12),
        ("highdim_big.txt",            gen_highdim,            500_000, 3, 20),
    ]

    for filename, generator, N, d, k in configs:
        print(f"\nGenerating {filename} ...")
        data = generator(N, d, k) if generator != gen_highdim or generator != gen_uniform else generator(N, d)
        write_text(OUT / filename, data, k)
        write_binary(OUT / (filename.replace(".txt", ".dat")), data, k)

    print("\nAll datasets created in:", OUT.resolve())
