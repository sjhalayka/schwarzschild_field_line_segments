#include "main.h"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <limits>

// Atomic counter for progress tracking
std::atomic<long long unsigned int> global_progress(0);



// Length of each line segment the ray is broken into
real_type segment_length = 1.0;

// Slab test for a finite line segment against an axis-aligned bounding box.
// Returns true when any part of the segment lies inside the box.
bool intersect_segment_AABB(
	const vector_3 min_location,
	const vector_3 max_location,
	const vector_3 segment_start,
	const vector_3 segment_end)
{

	vector_3 segment_mid_point;
	segment_mid_point.x = (segment_start.x + segment_end.x) * 0.5;
	segment_mid_point.y = (segment_start.y + segment_end.y) * 0.5;
	segment_mid_point.z = (segment_start.z + segment_end.z) * 0.5;




	return
		segment_mid_point.x >= min_location.x &&
		segment_mid_point.x <= max_location.x &&
		segment_mid_point.y >= min_location.y &&
		segment_mid_point.y <= max_location.y &&
		segment_mid_point.z >= min_location.z &&
		segment_mid_point.z <= max_location.z;







	//const vector_3 d = segment_end - segment_start;

	//const real_type s[3] = { segment_start.x, segment_start.y, segment_start.z };
	//const real_type dd[3] = { d.x, d.y, d.z };
	//const real_type mn[3] = { min_location.x, min_location.y, min_location.z };
	//const real_type mx[3] = { max_location.x, max_location.y, max_location.z };

	//// Segment parameter runs from 0 at the start to 1 at the end
	//real_type t0 = 0.0;
	//real_type t1 = 1.0;

	//for (size_t i = 0; i < 3; i++)
	//{
	//	if (fabs(dd[i]) < 1e-20)
	//	{
	//		// Segment is parallel to this pair of slabs
	//		if (s[i] < mn[i] || s[i] > mx[i])
	//			return false;

	//		continue;
	//	}

	//	const real_type inv = 1.0 / dd[i];

	//	real_type ta = (mn[i] - s[i]) * inv;
	//	real_type tb = (mx[i] - s[i]) * inv;

	//	if (ta > tb)
	//		swap(ta, tb);

	//	if (ta > t0)
	//		t0 = ta;

	//	if (tb < t1)
	//		t1 = tb;

	//	if (t0 > t1)
	//		return false;
	//}

	//return true;
}

// Breaks the ray into segments of segment_length and counts how many of
// those segments collide with the box. Every segment is tested directly --
// there is no line-box test anywhere in here.
//
// The march stops at the farthest corner of the box, since no point beyond
// that distance can possibly be inside it. It also stops early once the
// segments have entered and then left the box, because a box is convex and
// so the colliding segments form one contiguous run.
real_type intersect_AABB(
	const vector_3 min_location,
	const vector_3 max_location,
	vector_3 ray_origin, vector_3 ray_dir,
	const real_type emitter_radius,
	const real_type receiver_distance,
	const real_type receiver_distance_plus,
	const real_type receiver_radius)
{
	const real_type max_distance = receiver_distance_plus + receiver_radius * sqrt(3.0);

	vector_3 ray = ray_dir * segment_length;

	vector_3 segment_start = ray_origin;
	vector_3 segment_end = ray_origin + ray;

	real_type total_length = 0;

	while (segment_end.length() < max_distance)
	{
		if (intersect_segment_AABB(min_location, max_location, segment_start, segment_end))
			total_length += (segment_end - segment_start).length();

		segment_start = segment_end;
		segment_end += ray;
	}


	//const long long unsigned int segment_count =
	//	static_cast<long long unsigned int>(ceil(max_distance / segment_length));

	//real_type total_length = 0;
	//bool found_hit = false;

	//for (long long unsigned int i = 0; i < segment_count; i++)
	//{
	//	const vector_3 segment_start = ray_origin + ray_dir * (i * segment_length);
	//	const vector_3 segment_end = ray_origin + ray_dir * ((i + 1) * segment_length);

	//	if (intersect_segment_AABB(min_location, max_location, segment_start, segment_end))
	//	{
	//		if (false == found_hit)
	//		{
	//			tmin = i * segment_length;
	//			found_hit = true;
	//		}

	//		tmax = (i + 1) * segment_length;
	//		total_length += segment_length;
	//	}
	//	else if (found_hit)
	//	{
	//		break;
	//	}
	//}

	return total_length;
}

real_type intersect(
	const vector_3 location,
	const vector_3 normal,
	const real_type epsilon,
	const vector_3 min_location,
	const vector_3 max_location,
	const real_type emitter_radius,
	const real_type receiver_distance,
	const real_type receiver_distance_plus,
	const real_type receiver_radius)
{
	return intersect_AABB(
		min_location,
		max_location,
		location, normal,
		emitter_radius,
		receiver_distance,
		receiver_distance_plus,
		receiver_radius);
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

	vector_3 location(x, y, z);
	location.normalize();

	return location;
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
	const real_type epsilon,
	real_type& result_count,
	real_type& result_count_plus,
	real_type& result_count_forward)
{
	// Thread-local random number generator
	std::mt19937 local_gen(thread_seed);
	std::uniform_real_distribution<real_type> local_dis(0.0, 1.0);

	real_type local_count = 0;
	real_type local_count_plus = 0;
	real_type local_count_forward = 0;


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





		vector_3 aabb_min_location(-receiver_radius + receiver_distance, -receiver_radius, -receiver_radius);
		vector_3 aabb_max_location(receiver_radius + receiver_distance, receiver_radius, receiver_radius);


		vector_3 right_min_location = aabb_min_location;
		right_min_location.x += epsilon;

		vector_3 right_max_location = aabb_max_location;
		right_max_location.x += epsilon;

		vector_3 forward_min_location = aabb_min_location;
		forward_min_location.z += epsilon;

		vector_3 forward_max_location = aabb_max_location;
		forward_max_location.z += epsilon;



		local_count += intersect(
			location, normal,
			epsilon, aabb_min_location, aabb_max_location,
			emitter_radius,
			receiver_distance,
			receiver_distance_plus,
			receiver_radius
		);

		local_count_plus += intersect(
			location, normal,
			epsilon, right_min_location, right_max_location,
			emitter_radius,
			receiver_distance,
			receiver_distance_plus,
			receiver_radius
		);

		local_count_forward += intersect(
			location, normal,
			epsilon, forward_min_location, forward_max_location,
			emitter_radius,
			receiver_distance,
			receiver_distance_plus,
			receiver_radius);




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
	result_count_forward = local_count_forward;
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

pair<real_type, real_type> get_intersecting_line_density(
	const long long unsigned int n,
	const real_type emitter_radius,
	const real_type receiver_distance,
	const real_type receiver_distance_plus,
	const real_type receiver_radius,
	const real_type epsilon)
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
	std::vector<real_type> thread_counts_forward(num_threads, 0);

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
			epsilon,
			std::ref(thread_counts[t]),
			std::ref(thread_counts_plus[t]),
			std::ref(thread_counts_forward[t])
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
	real_type total_count_forward = 0;

	for (unsigned int t = 0; t < num_threads; t++)
	{
		total_count += thread_counts[t];
		total_count_plus += thread_counts_plus[t];
		total_count_forward += thread_counts_forward[t];
	}

	return pair<real_type, real_type>(total_count_plus - total_count, total_count_forward - total_count);
}

int main(int argc, char** argv)
{
	ofstream outfile_numerical("Schwarzschild_numerical");
	ofstream outfile_analytical("Schwarzschild_analytical");
	ofstream outfile_Newton("Newton_analytical");



	const real_type n_geometrized = 1e9; // field line count
	const real_type spin = 0.5; // a_*

	// --- derived ---
	const real_type spin_root = sqrt(1.0 - spin * spin);

	const real_type emitter_mass_geometrized =
		sqrt(n_geometrized * log(2.0) / (2.0 * pi * (1.0 + spin_root)));

	const real_type emitter_a_geometrized =
		spin * emitter_mass_geometrized;

	const real_type emitter_r_plus_geometrized =
		emitter_mass_geometrized * (1.0 + spin_root);

	const real_type emitter_area_geometrized =
		4.0 * n_geometrized * log(2.0);   // exact inverse of n = A / (4 ln2)

	//const real_type J = emitter_a_geometrized * emitter_mass_geometrized;



	const real_type receiver_radius_geometrized =
		emitter_r_plus_geometrized * 0.01; // Minimum one Planck unit



	real_type start_pos =
		emitter_r_plus_geometrized
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

		segment_length = receiver_radius_geometrized / 10.0;// epsilon / 10.0;

		// beta function
		const pair<real_type, real_type> collision_count_plus_minus_collision_count =
			get_intersecting_line_density(
				static_cast<long long unsigned int>(n_geometrized),
				emitter_r_plus_geometrized,
				receiver_distance_geometrized,
				receiver_distance_plus_geometrized,
				receiver_radius_geometrized,
				epsilon);

		// alpha variable
		const real_type gradient_integer =
			collision_count_plus_minus_collision_count.first
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


		const real_type dt_Schwarzschild = sqrt(1 - emitter_r_plus_geometrized / receiver_distance_geometrized);

		const real_type a_Schwarzschild_geometrized =
			emitter_r_plus_geometrized / (pi * pow(receiver_distance_geometrized, 2.0) * dt_Schwarzschild);

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