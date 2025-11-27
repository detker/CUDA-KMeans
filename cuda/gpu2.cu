#include "gpu2.cuh"

template<int D>
struct AssignAndCheckChangedFunctor
{
    int K;
    int N;
    double* points;
    double* centroids;
    unsigned char* assignments;
    AssignAndCheckChangedFunctor(double *points_ptr, double* centroids_ptr, unsigned char* assignments_ptr, int n, int k) : points(points_ptr), centroids(centroids_ptr), assignments(assignments_ptr), N(n), K(k) {}

    __host__ __device__
    thrust::tuple<int,int> operator()(int idx) const
    {
        // int base = idx * D;
		unsigned char best_cluster;
		double min_distance = DBL_MAX;
        for (int k = 0; k < K; ++k)
        {
            double sum = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                // double diff = points[base + d] - centroids[k * D + d];
                double diff = points[d * N + idx] - centroids[d * K + k];
                sum += diff * diff;
            }
            if(sum < min_distance)
            {
                min_distance = sum;
                best_cluster = k;
			}
        }

		int changed = (assignments[idx] != best_cluster) ? 1 : 0;

		return thrust::make_tuple(best_cluster, changed);
    }
};


template<int D>
void thrust_kmeans_host(const double* datapoints, double* centroids,
    int N, int K, unsigned char* assignments, TimerManager *tm)
{
    TimerGPU timerGPU;
    tm->SetTimer(&timerGPU);

    // double *datapoints_col_major = new double[N * D];
    double *centroids_col_major = new double[K * D];
    // row_to_col_major<double>(datapoints, datapoints_col_major, N, D);
    // row_to_col_major<double>(datapoints, centroids_col_major, K, D);
    // thrust::device_vector<double> d_datapoints(datapoints_col_major, datapoints_col_major + N * D);
    // thrust::device_vector<double> d_datapoints(datapoints_col_major, datapoints_col_major + N * D);
    thrust::device_vector<double> d_datapoints(datapoints, datapoints + N * D);
    // double *dp = thrust::raw_pointer_cast(d_datapoints.data());
    for (int k = 0; k < K; ++k)
    {
        #pragma unroll 4
        for (int d = 0; d < D; ++d)
        {
            centroids_col_major[d * K + k] = datapoints[d*N + k];
        }
    }
	thrust::device_vector<double> d_centroids(centroids_col_major, centroids_col_major + K * D);
    // thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(K * D),
    //     d_centroids.begin(),
    //     [=] __device__(int idx) {
    //         int row = idx / K;
    //         int col = idx % K;
    //         return dp[row * N + col];
    //     });

    thrust::device_vector<double> newCentroids(K*D, 0.0);
    thrust::device_vector<unsigned char> d_assignments(N, 0);
	thrust::device_vector<unsigned char> d_oldAssignments(N, 0);
    thrust::device_vector<int> d_assignmentChanged(N, 0);

    unsigned int delta = N;

    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; ++iter) {
        thrust::device_vector<int> ones(N, 1);
        thrust::device_vector<int> indices(N);
        thrust::sequence(indices.begin(), indices.end());

        auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_assignments.begin(), d_assignmentChanged.begin()));
        AssignAndCheckChangedFunctor<D> f(thrust::raw_pointer_cast(d_datapoints.data()), thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_assignments.data()), N, K);
        tm->Start();
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(N), zip, f);
        tm->Stop();

        tm->Start();
        delta = thrust::reduce(d_assignmentChanged.begin(), d_assignmentChanged.end());
        tm->Stop();

        d_oldAssignments = d_assignments;
        tm->Start();
        thrust::sort_by_key(d_assignments.begin(), d_assignments.end(), indices.begin());
        tm->Stop();

        thrust::device_vector<int> clustersIdxs(K);
		thrust::device_vector<int> clusterCounts(K, 0.0);
		thrust::device_vector<double> clusterSums(K, 0.0);
        double* points_ptr = thrust::raw_pointer_cast(d_datapoints.data());
        // double* newCentroids_ptr = thrust::raw_pointer_cast(newCentroids.data());

        #pragma unroll 4
        for (int d = 0; d < D; ++d)
        {
			thrust::device_vector<double> pointsAlongDim(N);
            tm->Start();
            thrust::transform(indices.begin(), indices.end(), pointsAlongDim.begin(),
                [=] __device__(int idx) {
                    return points_ptr[d * N + idx];
			});
            tm->Stop();

            auto binary_op = [=] __device__(thrust::tuple<double, int> idx1_tuple, thrust::tuple<double, int> idx2_tuple) {
				double val1 = thrust::get<0>(idx1_tuple);
				double val2 = thrust::get<0>(idx2_tuple);
				int count1 = thrust::get<1>(idx1_tuple);
				int count2 = thrust::get<1>(idx2_tuple);
                return thrust::make_tuple(val1+val2, count1+count2);
            };

            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(clusterSums.begin(), clusterCounts.begin()));
            auto zipped_in = thrust::make_zip_iterator(thrust::make_tuple(pointsAlongDim.begin(), ones.begin()));

            tm->Start();
            thrust::reduce_by_key(d_assignments.begin(), d_assignments.end(), zipped_in, clustersIdxs.begin(), zipped, thrust::equal_to<unsigned char>(), binary_op);
            tm->Stop();

            double* sums_ptr = thrust::raw_pointer_cast(clusterSums.data());
            int* counts_ptr = thrust::raw_pointer_cast(clusterCounts.data());
            int* cluster_ids_ptr = thrust::raw_pointer_cast(clustersIdxs.data());
            double* centroids_ptr = thrust::raw_pointer_cast(newCentroids.data());

            tm->Start();
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(K),
                [=] __device__(int idx) {
                int k = cluster_ids_ptr[idx];
                int count = counts_ptr[idx];
                if (count > 0)
                    centroids_ptr[d * K + k] = sums_ptr[idx] / (double)count;
                }
            );
            tm->Stop();
        }

		d_centroids = newCentroids;
        d_assignments = d_oldAssignments;

        printf("Iteration: %d, changes: %d\n", iter, delta);
    }

    // if (D == 3) 
    // {
    //     float minx, maxx, miny, maxy, minz, maxz;
    //     compute_bounds(datapoints, N, minx, maxx, miny, maxy, minz, maxz);
    //     // render(deviceDatapoints, deviceAssignments, N, K, minx, maxx, miny, maxy, minz, maxz);
    //     render(thrust::raw_pointer_cast(d_datapoints.data()), thrust::raw_pointer_cast(d_assignments.data()), N, K, minx, maxx, miny, maxy, minz, maxz);
    // }

	thrust::copy(d_assignments.begin(), d_assignments.end(), assignments);
    thrust::copy(d_centroids.begin(), d_centroids.end(), centroids_col_major);
    col_to_row_major<double>(centroids_col_major, centroids, K, D);    

    // free(datapoints_col_major);
    free(centroids_col_major);
}

// Type alias to reduce verbosity
using KMeansFunc = void(const double*, double*, int, int, unsigned char*, TimerManager*);

// Explicit template instantiations for dimensions 1-20
template KMeansFunc thrust_kmeans_host<1>;
template KMeansFunc thrust_kmeans_host<2>;
template KMeansFunc thrust_kmeans_host<3>;
template KMeansFunc thrust_kmeans_host<4>;
template KMeansFunc thrust_kmeans_host<5>;
template KMeansFunc thrust_kmeans_host<6>;
template KMeansFunc thrust_kmeans_host<7>;
template KMeansFunc thrust_kmeans_host<8>;
template KMeansFunc thrust_kmeans_host<9>;
template KMeansFunc thrust_kmeans_host<10>;
template KMeansFunc thrust_kmeans_host<11>;
template KMeansFunc thrust_kmeans_host<12>;
template KMeansFunc thrust_kmeans_host<13>;
template KMeansFunc thrust_kmeans_host<14>;
template KMeansFunc thrust_kmeans_host<15>;
template KMeansFunc thrust_kmeans_host<16>;
template KMeansFunc thrust_kmeans_host<17>;
template KMeansFunc thrust_kmeans_host<18>;
template KMeansFunc thrust_kmeans_host<19>;
template KMeansFunc thrust_kmeans_host<20>;