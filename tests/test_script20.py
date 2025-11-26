import numpy as np
from sklearn.datasets import make_moons, make_circles, load_iris, load_wine, load_breast_cancer, make_blobs
import struct
import subprocess
import os

SEED = 42
np.random.seed(SEED)

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
        # Write point coordinates
        for i, point in enumerate(data):
            f.write(str(i+1) + " " + " ".join(map(str, point)) + "\n")

    print(f"[saved] {filename} | shape=({N}, {d}), k={k}")


# -------------------------------------------------------------------
# Dataset generators - Original
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
# NEW Dataset generators
# -------------------------------------------------------------------
def gen_moons(N, d, k):
    """Non-linear clusters (moons shape) - challenging for k-means"""
    X, _ = make_moons(n_samples=N, noise=0.1, random_state=SEED)
    # Pad to d dimensions if needed
    if d > 2:
        padding = np.random.randn(N, d - 2) * 0.1
        X = np.hstack([X, padding])
    return X

def gen_circles(N, d, k):
    """Concentric circles - challenging for k-means"""
    X, _ = make_circles(n_samples=N, noise=0.05, factor=0.5, random_state=SEED)
    # Pad to d dimensions if needed
    if d > 2:
        padding = np.random.randn(N, d - 2) * 0.1
        X = np.hstack([X, padding])
    return X

def gen_elongated_clusters(N, d, k):
    """Elongated clusters with different variances"""
    centers = np.random.uniform(-5, 5, (k, d))
    pts_per_cluster = N // k
    data = []
    for i in range(k):
        # Create elongated clusters by scaling different dimensions differently
        scales = np.random.uniform(0.1, 2.0, d)
        pts = centers[i] + np.random.randn(pts_per_cluster, d) * scales
        data.append(pts)
    data = np.vstack(data)
    if data.shape[0] < N:
        extra = N - data.shape[0]
        data = np.vstack([data, centers[0] + np.random.randn(extra, d)])
    return data

def gen_sparse_dense_mix(N, d, k):
    """Mix of sparse and dense clusters"""
    centers = np.random.uniform(-10, 10, (k, d))
    dense_clusters = k // 2
    sparse_clusters = k - dense_clusters
    
    pts_dense = N // (dense_clusters + 2 * sparse_clusters)
    pts_sparse = pts_dense * 2
    
    data = []
    for i in range(dense_clusters):
        pts = centers[i] + np.random.randn(pts_dense, d) * 0.3
        data.append(pts)
    for i in range(dense_clusters, k):
        pts = centers[i] + np.random.randn(pts_sparse, d) * 2.0
        data.append(pts)
    
    data = np.vstack(data)
    if data.shape[0] < N:
        extra = N - data.shape[0]
        data = np.vstack([data, centers[0] + np.random.randn(extra, d) * 0.3])
    elif data.shape[0] > N:
        data = data[:N]
    return data

def gen_blobs_sklearn(N, d, k):
    """Well-separated blobs using sklearn"""
    X, _ = make_blobs(n_samples=N, n_features=d, centers=k, 
                      cluster_std=1.0, random_state=SEED)
    return X

def gen_grid_clusters(N, d, k):
    """Clusters arranged in a grid pattern (2D projection)"""
    side = int(np.sqrt(k))
    if side * side < k:
        side += 1
    
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= k:
                break
            center = [i * 5.0, j * 5.0] + [0.0] * (d - 2)
            centers.append(center)
    centers = np.array(centers[:k])
    
    pts_per_cluster = N // k
    data = []
    for i in range(k):
        pts = centers[i] + np.random.randn(pts_per_cluster, d) * 0.5
        data.append(pts)
    data = np.vstack(data)
    if data.shape[0] < N:
        extra = N - data.shape[0]
        data = np.vstack([data, centers[0] + np.random.randn(extra, d) * 0.5])
    return data


# -------------------------------------------------------------------
# Real dataset loaders
# -------------------------------------------------------------------
def load_iris_data():
    """Load Iris dataset (150 samples, 4 features, 3 classes)"""
    iris = load_iris()
    return iris.data, 3

def load_wine_data():
    """Load Wine dataset (178 samples, 13 features, 3 classes)"""
    wine = load_wine()
    return wine.data, 3

def load_breast_cancer_data():
    """Load Breast Cancer dataset (569 samples, 30 features, 2 classes)"""
    bc = load_breast_cancer()
    return bc.data, 2

def scale_real_dataset(data, N):
    """Scale a real dataset by repeating it to reach N samples"""
    current_N = data.shape[0]
    if N <= current_N:
        return data[:N]
    
    repeats = (N // current_N) + 1
    scaled = np.tile(data, (repeats, 1))
    # Add small noise to repeated samples
    noise = np.random.randn(*scaled.shape) * 0.01
    scaled = scaled + noise
    return scaled[:N]


# -------------------------------------------------------------------
# Testing utilities
# -------------------------------------------------------------------
def read_output_mine(filename):
    clusters = []
    assignments = []
    with open(filename, 'r') as f:
        for line in f:
            point = list(map(float, line.strip().split(' ')))
            if len(point) == 1:
                assignments.append(int(point[0]))
            else:
                clusters.append(point)
    return clusters, assignments

def launch_mine(data_format, compute_method, input_file, output_file):
    cmd = ['../build/KMeans',
            data_format,
            compute_method,
            input_file,
            output_file]
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

def run_test(filename, data, k, OUT='../data/'):
    """Run a single test comparing GPU and CPU implementations"""
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print(f"{'='*60}")
    
    # Write data
    write_binary(OUT + filename, data, k)
    
    # Run GPU version
    print("Running GPU version...")
    launch_mine('bin', 'gpu1', OUT + filename, OUT + filename + '_output_gpu.txt')
    print('✓ KMeans GPU finished')
    
    # Run CPU version
    print("Running CPU version...")
    launch_mine('bin', 'cpu', OUT + filename, OUT + filename + '_output_cpu.txt')
    print('✓ KMeans CPU finished')
    
    # Compare results
    clusters_gpu, assignments_gpu = read_output_mine(OUT + filename + '_output_gpu.txt')
    clusters_cpu, assignments_cpu = read_output_mine(OUT + filename + '_output_cpu.txt')
    compare_results(clusters_gpu, assignments_gpu, clusters_cpu, assignments_cpu)
    print(f'✓ Test PASSED: {filename}')
    print(f"  - Data shape: {data.shape}")
    print(f"  - Clusters: {k}")
    print(f"  - GPU clusters found: {len(clusters_gpu)}")
    print(f"  - CPU clusters found: {len(clusters_cpu)}")


# -------------------------------------------------------------------
# Main testing script
# -------------------------------------------------------------------
if __name__ == "__main__":
    OUT = '../data/'
    os.makedirs(OUT, exist_ok=True)
    
    # Test configurations
    test_suites = {
        "quick": [
            # Small synthetic tests
            ("small_uniform.dat", gen_uniform, 1000, 3, 5),
            ("small_gaussian.dat", gen_gaussian_clusters, 2000, 3, 4),
        ],
        
        "real_datasets": [
            # Real datasets
            ("iris_original.dat", lambda N, d, k: load_iris_data()[0], 150, 4, 3),
            ("iris_scaled_10k.dat", lambda N, d, k: scale_real_dataset(load_iris_data()[0], N), 10000, 4, 3),
            ("wine_original.dat", lambda N, d, k: load_wine_data()[0], 178, 13, 3),
            ("wine_scaled_50k.dat", lambda N, d, k: scale_real_dataset(load_wine_data()[0], N), 50000, 13, 3),
            # ("breast_cancer.dat", lambda N, d, k: load_breast_cancer_data()[0], 569, 30, 2),
        ],
        
        "synthetic_medium": [
            # Medium-sized synthetic tests
            ("uniform_10k.dat", gen_uniform, 10000, 3, 5),
            ("gaussian_20k.dat", gen_gaussian_clusters, 20000, 4, 8),
            ("overlap_15k.dat", gen_overlapping_clusters, 15000, 3, 6),
            ("skewed_30k.dat", gen_skewed_density, 30000, 3, 10),
            ("blobs_25k.dat", gen_blobs_sklearn, 25000, 5, 7),
        ],
        
        "challenging": [
            # Challenging patterns for k-means
            ("moons_5k.dat", gen_moons, 5000, 3, 2),
            ("circles_8k.dat", gen_circles, 8000, 3, 2),
            ("elongated_12k.dat", gen_elongated_clusters, 12000, 4, 6),
            ("sparse_dense_20k.dat", gen_sparse_dense_mix, 20000, 3, 8),
            ("grid_10k.dat", gen_grid_clusters, 10000, 4, 9),
        ],
        
        "high_dimensional": [
            # High-dimensional tests
            ("highdim_10d.dat", gen_highdim, 5000, 10, 5),
            ("highdim_20d.dat", gen_highdim, 3000, 20, 4),
            # ("highdim_50d.dat", gen_highdim, 2000, 50, 6),
            ("gaussian_highdim.dat", gen_gaussian_clusters, 10000, 15, 8),
        ],
        
        "large_scale": [
            # Large-scale tests (original size)
            ("uniform_big.dat", gen_uniform, 1200000, 3, 20),
            ("overlap_big.dat", gen_overlapping_clusters, 900000, 3, 20),
            ("gaussian_big.dat", gen_gaussian_clusters, 1000000, 3, 20),
            ("skewed_big.dat", gen_skewed_density, 1500000, 3, 20),
        ],
        
        "extreme": [
            # Extreme cases
            ("many_clusters.dat", gen_gaussian_clusters, 50000, 3, 20),
            ("very_highdim.dat", gen_highdim, 10000, 20, 20),
            ("tiny_dataset.dat", gen_uniform, 100, 2, 20),
        ],

        # "ultra extreme": [
        #     # Ultra Extreme cases
        #     ("many_clusters.dat", gen_gaussian_clusters, 50_000_000, 3, 8)
        #     #("very_highdim.dat", gen_highdim, 50_000_00, 3, 10)
        # ]

#        "gpu (cpu rather...) go brr": [
#            ("very_highdim.dat", gen_highdim, 50_000_000, 20, 20)
#        ]
    }
    
    # Select which test suite to run
    print("Available test suites:")
    for i, suite_name in enumerate(test_suites.keys(), 1):
        print(f"  {i}. {suite_name} ({len(test_suites[suite_name])} tests)")
    print(f"  {len(test_suites)+1}. all")
    
    choice = input("\nSelect test suite (number or name): ").strip().lower()
    
    # Parse choice
    if choice == "all" or choice == str(len(test_suites)+1):
        selected_suites = list(test_suites.keys())
    elif choice.isdigit() and 1 <= int(choice) <= len(test_suites):
        selected_suites = [list(test_suites.keys())[int(choice)-1]]
    elif choice in test_suites:
        selected_suites = [choice]
    else:
        print("Invalid choice, running 'quick' suite")
        selected_suites = ["quick"]
    
    # Run selected test suites
    total_tests = sum(len(test_suites[s]) for s in selected_suites)
    passed_tests = 0
    failed_tests = 0
    
    print(f"\n{'#'*60}")
    print(f"Running {total_tests} tests from {len(selected_suites)} suite(s)")
    print(f"{'#'*60}\n")
    
    for suite_name in selected_suites:
        print(f"\n{'*'*60}")
        print(f"TEST SUITE: {suite_name.upper()}")
        print(f"{'*'*60}")
        
        configs = test_suites[suite_name]
        
        for filename, generator, N, d, k in configs:
            try:
                print(f"\nGenerating data for {filename}...")
                if callable(generator):
                    data = generator(N, d, k)
                else:
                    data = generator
                
                run_test(filename, data, k, OUT)
                passed_tests += 1
                
            except Exception as e:
                print(f"\n✗ Test FAILED: {filename}")
                print(f"  Error: {str(e)}")
                failed_tests += 1
    
    # Summary
    print(f"\n{'#'*60}")
    print(f"TEST SUMMARY")
    print(f"{'#'*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests} ✓")
    print(f"Failed: {failed_tests} ✗")
    print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    print(f"{'#'*60}\n")
