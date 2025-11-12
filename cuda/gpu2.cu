#include "gpu2.cuh"


typedef struct AssignAndCheckChangedFunctor
{
    int D;
    int K;
    double* points;
    double* centroids;
    int* assignments;
    AssignAndCheckChangedFunctor(double *points_ptr, double* centroids_ptr, int* assignments_ptr, int d, int k) : points(points_ptr), centroids(centroids_ptr), assignments(assignments_ptr), D(d), K(k) {}

    __host__ __device__
    thrust::tuple<int,int> operator()(int idx) const
    {
        int base = idx * D;
		int best_cluster = 0;
		double min_distance = DBL_MAX;
        for (int k = 0; k < K; ++k)
        {
            double sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                double diff = points[base + d] - centroids[k * D + d];
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
} AssignAndCheckChangedFunctor;


void thrust_kmeans_host(double* datapoints, double* centroids,
    int N, int K, int D, int* assignments, TimerManager *tm)
{
    TimerGPU timerGPU;
    tm->SetTimer(&timerGPU);

    thrust::device_vector<double> d_datapoints(datapoints, datapoints + N * D);
	thrust::device_vector<double> d_centroids(datapoints, datapoints + K * D);
    thrust::device_vector<int> d_assignments(N, 0);
	thrust::device_vector<int> d_oldAssignments(N, 0);
    thrust::device_vector<int> d_assignmentChanged(N, 0);

    unsigned int delta = N;

    for (int iter = 0; iter < MAX_ITERATIONS && delta > 0; ++iter) {
        thrust::device_vector<int> ones(N, 1);
        thrust::device_vector<int> indices(N);
        thrust::sequence(indices.begin(), indices.end());

        auto zip = thrust::make_zip_iterator(thrust::make_tuple(d_assignments.begin(), d_assignmentChanged.begin()));
        AssignAndCheckChangedFunctor f(thrust::raw_pointer_cast(d_datapoints.data()), thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_assignments.data()), D, K);
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

        thrust::device_vector<double> newCentroids(K*D, 0.0);
        thrust::device_vector<int> clustersIdxs(K);
		thrust::device_vector<int> clusterCounts(K, 0.0);
		thrust::device_vector<double> clusterSums(K, 0.0);
        double* points_ptr = thrust::raw_pointer_cast(d_datapoints.data());
        double* newCentroids_ptr = thrust::raw_pointer_cast(newCentroids.data());


        for (int d = 0; d < D; ++d)
        {
			thrust::device_vector<double> pointsAlongDim(N);
            tm->Start();
            thrust::transform(indices.begin(), indices.end(), pointsAlongDim.begin(),
                [=] __device__(int idx) {
                    return points_ptr[idx * D + d];
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
            thrust::reduce_by_key(d_assignments.begin(), d_assignments.end(), zipped_in, clustersIdxs.begin(), zipped, thrust::equal_to<int>(), binary_op);
            tm->Stop();

            double* sums_ptr = thrust::raw_pointer_cast(clusterSums.data());
            int* counts_ptr = thrust::raw_pointer_cast(clusterCounts.data());
            double* centroids_ptr = thrust::raw_pointer_cast(newCentroids.data());

            tm->Start();
            thrust::for_each(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(K),
                [=] __device__(int k) {
                int count = counts_ptr[k];
                if (count > 0)
                    centroids_ptr[k * D + d] = sums_ptr[k] / (double)count;
                else
                    centroids_ptr[k * D + d] = 0.0;
                }
            );
            tm->Stop();
        }


		d_centroids = newCentroids;
        d_assignments = d_oldAssignments;

        printf("Iteration: %d, changes: %d\n", iter, delta);
    }
	thrust::copy(d_assignments.begin(), d_assignments.end(), assignments);
    thrust::copy(d_centroids.begin(), d_centroids.end(), centroids);

    //if (D == 3) render(datapoints, d_datapoints, d_assignments, N, K);
}