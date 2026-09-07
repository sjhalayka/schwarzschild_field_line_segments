#include "main.h"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>

// Atomic counter for progress tracking
std::atomic<long long unsigned int> global_progress(0);

//
//real_type intersect_AABB(const vector_3 min_location, const vector_3 max_location, const vector_3 ray_origin, const vector_3 ray_dir, real_type& tmin, real_type& tmax)
//{
//	tmin = (min_location.x - ray_origin.x) / ray_dir.x;
//	tmax = (max_location.x - ray_origin.x) / ray_dir.x;
//
//	if (tmin > tmax)
//		swap(tmin, tmax);
//
//	real_type tymin = (min_location.y - ray_origin.y) / ray_dir.y;
//	real_type tymax = (max_location.y - ray_origin.y) / ray_dir.y;
//
//	if (tymin > tymax)
//		swap(tymin, tymax);
//
//	if ((tmin > tymax) || (tymin > tmax))
//		return 0;
//
//	if (tymin > tmin)
//		tmin = tymin;
//
//	if (tymax < tmax)
//		tmax = tymax;
//
//	real_type tzmin = (min_location.z - ray_origin.z) / ray_dir.z;
//	real_type tzmax = (max_location.z - ray_origin.z) / ray_dir.z;
//
//	if (tzmin > tzmax)
//		swap(tzmin, tzmax);
//
//	if ((tmin > tzmax) || (tzmin > tmax))
//		return 0;
//
//	if (tzmin > tmin)
//		tmin = tzmin;
//
//	if (tzmax < tmax)
//		tmax = tzmax;
//
//	if (tmin < 0 || tmax < 0)
//		return 0;
//
//	vector_3 ray_hit_start = ray_origin;
//	ray_hit_start.x += ray_dir.x * tmin;
//	ray_hit_start.y += ray_dir.y * tmin;
//	ray_hit_start.z += ray_dir.z * tmin;
//
//	vector_3 ray_hit_end = ray_origin;
//	ray_hit_end.x += ray_dir.x * tmax;
//	ray_hit_end.y += ray_dir.y * tmax;
//	ray_hit_end.z += ray_dir.z * tmax;
//
//	real_type l = (ray_hit_end - ray_hit_start).length();
//
//	return l;
//}

inline bool point_in_AABB(const vector_3 min_location, const vector_3 max_location, const vector_3 p)
{
	return p.x >= min_location.x && p.x <= max_location.x
		&& p.y >= min_location.y && p.y <= max_location.y
		&& p.z >= min_location.z && p.z <= max_location.z;
}

// Segment-marching alternative to intersect_AABB.
// Walks from ray_origin along ray_dir (assumed unit length) in steps of
// segment_length, up to march_distance, and sums the length of every
// segment that lies entirely inside the box (both endpoints inside; the box
// is convex, so that implies the whole segment is inside).
// tmin/tmax are set to the parametric start/end of the first and last
// interior segments, mirroring intersect_AABB.
real_type intersect_AABB_segments(
	const vector_3 min_location, const vector_3 max_location,
	const vector_3 ray_origin, const vector_3 ray_dir,
	const real_type segment_length, const real_type march_distance,
	real_type& tmin, real_type& tmax)
{
	tmin = tmax = 0;

	const long long unsigned int num_segments =
		static_cast<long long unsigned int>(ceil(march_distance / segment_length));

	real_type length_inside = 0;
	bool found = false;

	vector_3 seg_start = ray_origin;
	bool start_inside = point_in_AABB(min_location, max_location, seg_start);

	for (long long unsigned int s = 0; s < num_segments; s++)
	{
		const real_type t_end = (s + 1) * segment_length;

		vector_3 seg_end = ray_origin;
		seg_end.x += ray_dir.x * t_end;
		seg_end.y += ray_dir.y * t_end;
		seg_end.z += ray_dir.z * t_end;

		const bool end_inside = point_in_AABB(min_location, max_location, seg_end);

		if (start_inside && end_inside)
		{
			length_inside += segment_length;

			if (!found)
			{
				tmin = s * segment_length;
				found = true;
			}

			tmax = t_end;
		}
		else if (found)
		{
			// Convex box: once we've been inside and then leave, we never re-enter.
			break;
		}

		seg_start = seg_end;
		start_inside = end_inside;
	}

	return length_inside;
}

real_type intersect(
	const vector_3 location,
	const vector_3 normal,
	const real_type receiver_distance,
	const real_type receiver_radius)
{
	//const vector_3 circle_origin(receiver_distance, 0, 0);

	//if (normal.dot(circle_origin) <= 0)
	//	return 0.0;

	vector_3 min_location(-receiver_radius + receiver_distance, -receiver_radius, -receiver_radius);
	vector_3 max_location(receiver_radius + receiver_distance, receiver_radius, receiver_radius);

	real_type tmin = 0, tmax = 0;

	// March in steps of a fraction of the receiver radius. The furthest any
	// point of the box can be from the ray origin (which sits on the emitter
	// sphere of radius |location|) is bounded by |location| + distance to the
	// far corner, so marching that far guarantees we pass through the box.
	const real_type segment_length = receiver_radius * 0.01;
	const real_type march_distance =
		location.length() + receiver_distance + receiver_radius * sqrt(3.0);

	return intersect_AABB_segments(
		min_location, max_location, location, normal,
		segment_length, march_distance, tmin, tmax);
}

// Thread-local versions of random functions that take generator and distribution as parameters
vector_3 random_cosine_weighted_hemisphere(vector_3 normal,
	std::mt19937& local_gen, std::uniform_real_distribution<real_type>& local_dis)
{
	vector_3 r = vector_3(local_dis(local_gen), local_dis(local_gen), 0.0);
	vector_3 uu = normal.cross(vector_3(0.0, 1.0, 1.0)).normalize();
	vector_3 vv = uu.cross(normal);

	double ra = sqrt(r.y);
	double rx = ra * cos(2.0 * pi * r.x);
	double ry = ra * sin(2.0 * pi * r.x);
	double rz = sqrt(1.0 - r.y);
	vector_3 rr = vector_3(uu * rx + vv * ry + normal * rz);

	return rr.normalize();
}

vector_3 random_unit_vector(std::mt19937& local_gen, std::uniform_real_distribution<real_type>& local_dis)
{
	const real_type z = local_dis(local_gen) * 2.0 - 1.0;
	const real_type a = local_dis(local_gen) * 2.0 * pi;

	const real_type r = sqrt(1.0f - z * z);
	const real_type x = r * cos(a);
	const real_type y = r * sin(a);

	return vector_3(x, y, z).normalize();
}

// Worker function for each thread
void worker_thread(
	long long unsigned int start_idx,
	long long unsigned int end_idx,
	unsigned int thread_seed,
	const real_type emitter_radius,
	const real_type receiver_distance,
	const real_type receiver_distance_plus,
	const real_type receiver_radius,
	real_type& result_count,
	real_type& result_count_plus)
{
	// Thread-local random number generator
	std::mt19937 local_gen(thread_seed);
	std::uniform_real_distribution<real_type> local_dis(0.0, 1.0);

	real_type local_count = 0;
	real_type local_count_plus = 0;

	// Update progress every N iterations to reduce atomic overhead
	const long long unsigned int progress_update_interval = 10000;
	long long unsigned int local_progress = 0;

	for (long long unsigned int i = start_idx; i < end_idx; i++)
	{
		vector_3 location = random_unit_vector(local_gen, local_dis);

		location.x *= emitter_radius;
		location.y *= emitter_radius;
		location.z *= emitter_radius;

		vector_3 surface_normal = location;
		surface_normal.normalize();


		// A) Newtonian gravitation
		//vector_3 normal =
		//	surface_normal;

		// B) Schwarzschild gravitation, classical
		//vector_3 normal = 
		//	random_cosine_weighted_hemisphere(
		//		surface_normal, local_gen, local_dis);

		// C) Schwarzschild gravitation, quantum
		vector_3 r = random_unit_vector(local_gen, local_dis);

		r.x *= emitter_radius;
		r.y *= emitter_radius;
		r.z *= emitter_radius;

		vector_3 normal = (location - r).normalize();

		local_count += intersect(
			location, normal,
			receiver_distance, receiver_radius);

		local_count_plus += intersect(
			location, normal,
			receiver_distance_plus, receiver_radius);

		// Update global progress periodically
		local_progress++;
		if (local_progress >= progress_update_interval)
		{
			global_progress.fetch_add(local_progress, std::memory_order_relaxed);
			local_progress = 0;
		}
	}

	// Add any remaining progress
	if (local_progress > 0)
	{
		global_progress.fetch_add(local_progress, std::memory_order_relaxed);
	}

	result_count = local_count;
	result_count_plus = local_count_plus;
}

// Progress monitor function that runs on main thread
void progress_monitor(long long unsigned int total_iterations, std::atomic<bool>& done)
{
	auto start_time = std::chrono::steady_clock::now();

	while (!done.load(std::memory_order_relaxed))
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(500));

		long long unsigned int current = global_progress.load(std::memory_order_relaxed);
		double progress = static_cast<double>(current) / static_cast<double>(total_iterations);

		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

		// Estimate time remaining
		double eta_seconds = 0;
		if (progress > 0.001)
		{
			eta_seconds = (elapsed / progress) * (1.0 - progress);
		}

		cout << "\rProgress: " << fixed << (progress * 100.0) << "% "
			<< "| Elapsed: " << elapsed << "s "
			<< "| ETA: " << static_cast<int>(eta_seconds) << "s    " << flush;
	}

	cout << "\rProgress: 100.00% | Complete!                              " << endl;
}

real_type get_intersecting_line_density(
	const long long unsigned int n,
	const real_type emitter_radius,
	const real_type receiver_distance,
	const real_type receiver_distance_plus,
	const real_type receiver_radius)
{
	// Reset global progress counter
	global_progress.store(0, std::memory_order_relaxed);

	// Get number of hardware threads
	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) num_threads = 4; // Fallback if detection fails

	cout << "Using " << num_threads << " threads for " << n << " iterations" << endl;

	std::vector<std::thread> threads;
	std::vector<real_type> thread_counts(num_threads, 0);
	std::vector<real_type> thread_counts_plus(num_threads, 0);

	// Flag to signal progress monitor to stop
	std::atomic<bool> done(false);

	// Start progress monitor thread
	std::thread monitor_thread(progress_monitor, n, std::ref(done));

	// Calculate work distribution
	long long unsigned int iterations_per_thread = n / num_threads;
	long long unsigned int remainder = n % num_threads;

	long long unsigned int current_start = 0;

	for (unsigned int t = 0; t < num_threads; t++)
	{
		long long unsigned int thread_iterations = iterations_per_thread;
		if (t < remainder) thread_iterations++; // Distribute remainder

		long long unsigned int thread_end = current_start + thread_iterations;

		// Each thread gets a different seed based on thread index
		unsigned int thread_seed = t;

		threads.emplace_back(
			worker_thread,
			current_start,
			thread_end,
			thread_seed,
			emitter_radius,
			receiver_distance,
			receiver_distance_plus,
			receiver_radius,
			std::ref(thread_counts[t]),
			std::ref(thread_counts_plus[t])
		);

		current_start = thread_end;
	}

	// Wait for all worker threads to complete
	for (auto& t : threads)
	{
		t.join();
	}

	// Signal monitor thread to stop and wait for it
	done.store(true, std::memory_order_relaxed);
	monitor_thread.join();

	// Aggregate results
	real_type total_count = 0;
	real_type total_count_plus = 0;

	for (unsigned int t = 0; t < num_threads; t++)
	{
		total_count += thread_counts[t];
		total_count_plus += thread_counts_plus[t];
	}

	return total_count_plus - total_count;
}

int main(int argc, char** argv)
{
	ofstream outfile_numerical("Schwarzschild_numerical");
	ofstream outfile_analytical("Schwarzschild_analytical");
	ofstream outfile_Newton("Newton_analytical");

	const real_type emitter_radius_geometrized =
		sqrt(1e9 * log(2.0) / pi);

	const real_type receiver_radius_geometrized =
		emitter_radius_geometrized * 0.01; // Minimum one Planck unit

	const real_type emitter_area_geometrized =
		4.0 * pi
		* emitter_radius_geometrized
		* emitter_radius_geometrized;

	// Field line count
	const real_type n_geometrized =
		emitter_area_geometrized
		/ (log(2.0) * 4.0);

	const real_type emitter_mass_geometrized =
		emitter_radius_geometrized
		/ 2.0;

	real_type start_pos =
		emitter_radius_geometrized
		+ receiver_radius_geometrized;

	real_type end_pos = start_pos * 2.0;


	const size_t pos_res = 30; // Minimum 2 steps

	const real_type pos_step_size =
		(end_pos - start_pos)
		/ (pos_res - 1);

	const real_type epsilon =
		receiver_radius_geometrized;


	for (size_t i = 0; i < pos_res; i++)
	{
		cout << "\n=== Step " << (i + 1) << " of " << pos_res << " ===" << endl;

		const real_type receiver_distance_geometrized =
			start_pos + i * pos_step_size;

		const real_type receiver_distance_plus_geometrized =
			receiver_distance_geometrized + epsilon;

		// beta function
		const real_type collision_count_plus_minus_collision_count =
			get_intersecting_line_density(
				static_cast<long long unsigned int>(n_geometrized),
				emitter_radius_geometrized,
				receiver_distance_geometrized,
				receiver_distance_plus_geometrized,
				receiver_radius_geometrized);

		// alpha variable
		const real_type gradient_integer =
			collision_count_plus_minus_collision_count
			/ epsilon;

		// g variable
		real_type gradient_strength =
			-gradient_integer
			/
			(2.0 * receiver_radius_geometrized
				* receiver_radius_geometrized
				* receiver_radius_geometrized);

		const real_type a_Newton_geometrized =
			sqrt(
				n_geometrized * log(2.0)
				/
				(4.0 * pi *
					pow(receiver_distance_geometrized, 4.0))
			);

		const real_type a_flat_geometrized =
			gradient_strength * receiver_distance_geometrized * log(2)
			/ (8.0 * emitter_mass_geometrized);


		const real_type dt_Schwarzschild = sqrt(1 - emitter_radius_geometrized / receiver_distance_geometrized);

		const real_type a_Schwarzschild_geometrized =
			emitter_radius_geometrized / (pi * pow(receiver_distance_geometrized, 2.0) * dt_Schwarzschild);

		cout << "a_Schwarzschild_geometrized " << a_Schwarzschild_geometrized << endl;
		cout << "a_Newton_geometrized " << a_Newton_geometrized << endl;
		cout << "a_flat_geometrized " << a_flat_geometrized << endl;
		cout << a_Schwarzschild_geometrized / a_flat_geometrized << endl;
		cout << endl;
		cout << a_Newton_geometrized / a_flat_geometrized << endl;
		cout << endl << endl;


		outfile_numerical << receiver_distance_geometrized << " " << a_flat_geometrized << endl;
		outfile_analytical << receiver_distance_geometrized << " " << a_Schwarzschild_geometrized << endl;
		outfile_Newton << receiver_distance_geometrized << " " << a_Newton_geometrized << endl;


	}


}