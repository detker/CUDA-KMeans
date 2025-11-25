import numpy as np
import struct

def write_gaussian_stream(filename, N, d, k, spread=0.1, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, (k, d))

    with open(filename, "wb") as f:
        f.write(struct.pack("<iii", N, d, k))

        pts_per_cluster = N // k
        count = 0

        for i in range(k):
            num = pts_per_cluster if i < k - 1 else (N - count)
            pts = centers[i] + rng.standard_normal((num, d)) * spread
            f.write(pts.astype(np.float64).tobytes())
            count += num

    print(f"[saved] {filename} | shape=({N}, {d}), k={k}")

write_gaussian_stream("../data/gaussian_extreme_case.dat", 50_000_000, 20, 20)