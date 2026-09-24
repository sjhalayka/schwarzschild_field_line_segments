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

// Oriented bounding box.
// The box is centred on 'center'; its three local axes are orthonormal and
// 'half_extent' gives the half-size of the box along each local axis
// (half_extent.x along axis[0], .y along axis[1], .z along axis[2]).
struct OBB
{
	vector_3 center;
	vector_3 axis[3];
	vector_3 half_extent;
};

inline real_type dot3(const vector_3& a, const vector_3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Spherical unit vectors at the angles (theta, phi):
//   theta: polar angle measured from +z
//   phi:   azimuthal (equatorial) angle measured from +x toward +y
//
//   r_hat     = ( sin(t)cos(p),  sin(t)sin(p),  cos(t) )
//   theta_hat = ( cos(t)cos(p),  cos(t)sin(p), -sin(t) )
//   phi_hat   = (      -sin(p),        cos(p),       0 )
//
// (r_hat, theta_hat, phi_hat) is orthonormal and right-handed:
// r_hat x theta_hat = phi_hat.
void spherical_basis(
	const real_type theta,
	const real_type phi,
	vector_3& r_hat,
	vector_3& theta_hat,
	vector_3& phi_hat)
{
	const real_type st = sin(theta), ct = cos(theta);
	const real_type sp = sin(phi), cp = cos(phi);

	r_hat = vector_3(st * cp, st * sp, ct);
	theta_hat = vector_3(ct * cp, ct * sp, -st);
	phi_hat = vector_3(-sp, cp, 0.0);
}

// Build an OBB entirely from spherical coordinates (r, theta, phi).
//
// Position: the centre sits at the point (r, theta, phi),
//   center = r * r_hat(theta, phi).
//
// Orientation: the box's local axes are the spherical unit vectors at that
// same point, so the box always faces the origin:
//   axis[0] = r_hat      (radial,     half-size half_extent.x)
//   axis[1] = theta_hat  (polar,      half-size half_extent.y)
//   axis[2] = phi_hat    (azimuthal,  half-size half_extent.z)
//
// With theta = pi/2 and phi = 0 the centre is (r, 0, 0) and the axes are
// +x, -z, +y. For a cube (equal half-extents) this is exactly the original
// axis-aligned box.
OBB make_OBB_spherical(
	const real_type r,
	const real_type theta,
	const real_type phi,
	const vector_3 half_extent)
{
	vector_3 r_hat, theta_hat, phi_hat;
	spherical_basis(theta, phi, r_hat, theta_hat, phi_hat);

	OBB box;
	box.center = r_hat * r;
	box.half_extent = half_extent;
	box.axis[0] = r_hat;
	box.axis[1] = theta_hat;
	box.axis[2] = phi_hat;

	return box;
}

// Test a finite line segment against an oriented bounding box.
// As in the original AABB version, the segment counts as inside when its
// midpoint lies inside the box. The midpoint is projected onto each of the
// box's local axes and compared with the half-extent along that axis.
bool intersect_segment_OBB(
	const OBB& box,
	const vector_3 segment_start,
	const vector_3 segment_end)
{
	vector_3 segment_mid_point;
	segment_mid_point.x = (segment_start.x + segment_end.x) * 0.5;
	segment_mid_point.y = (segment_start.y + segment_end.y) * 0.5;
	segment_mid_point.z = (segment_start.z + segment_end.z) * 0.5;

	const vector_3 d = segment_mid_point - box.center;

	return
		fabs(dot3(d, box.axis[0])) <= box.half_extent.x &&
		fabs(dot3(d, box.axis[1])) <= box.half_extent.y &&
		fabs(dot3(d, box.axis[2])) <= box.half_extent.z;
}

real_type intersect_OBB(
	const OBB& box,
	vector_3 ray_origin, vector_3 ray_dir,
	const real_type emitter_radius,
	real_type emitter_spin_a, real_type emitter_mass)
{
	// No point of the box is farther from the origin than
	// |center| + |half-extent diagonal|, whatever the orientation.
	// One extra segment length ensures a segment whose midpoint is still
	// inside the box is never cut off by the loop condition.
	const real_type max_distance =
		box.center.length() + box.half_extent.length() * sqrt(3);// +segment_length;

	vector_3 ray = ray_dir * segment_length;

	vector_3 segment_start = ray_origin;
	vector_3 segment_end = ray_origin + ray;

	real_type total_length = 0;

	while (segment_end.length() < max_distance)
	{
		if (intersect_segment_OBB(box, segment_start, segment_end))
			total_length += (segment_end - segment_start).length();

		segment_start = segment_end;
		segment_end += ray;
	}

	return total_length;
}



real_type intersect(
	const vector_3 location,
	const vector_3 normal,
	const OBB& box,
	const real_type emitter_radius,
	real_type emitter_spin_a, real_type emitter_mass)
{
	return intersect_OBB(
		box,
		location, normal,
		emitter_radius,
		emitter_spin_a, emitter_mass);
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
	//const real_type receiver_distance_plus,
	const real_type receiver_radius,
	const real_type epsilon,
	const real_type emitter_spin_a,
	const real_type emitter_mass,
	const real_type receiver_theta,
	const real_type receiver_phi,
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

	// Receiver boxes, all defined in spherical coordinates (r, theta, phi).
	// They do not change between iterations, so they are built once per thread.
	// Each box is centred at its (r, theta, phi) point and oriented along the
	// local spherical axes there (see make_OBB_spherical).
	const vector_3 receiver_half_extent(receiver_radius, receiver_radius, receiver_radius);

	const OBB receiver_box =
		make_OBB_spherical(receiver_distance, receiver_theta, receiver_phi, receiver_half_extent);

	// Radial finite difference: step r by epsilon.
	const OBB right_box =
		make_OBB_spherical(receiver_distance + epsilon, receiver_theta, receiver_phi, receiver_half_extent);

	// Sideways (azimuthal) finite difference: step phi so that the centre moves
	// an arc length of epsilon along phi_hat, keeping r and theta fixed.
	// The box stays at distance receiver_distance and re-orients to the new
	// local spherical axes. (At the poles, sin(theta) = 0 and phi is undefined,
	// so fall back to a step in theta.)
	const real_type sin_theta = sin(receiver_theta);

	const real_type dphi = epsilon;// / (receiver_distance * 10);
	const OBB forward_box =
		make_OBB_spherical(receiver_distance, receiver_theta, receiver_phi + dphi, receiver_half_extent);

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
			receiver_box,
			emitter_radius,
			emitter_spin_a, emitter_mass);

		local_count_plus += intersect(
			location, normal,
			right_box,
			emitter_radius,
			emitter_spin_a, emitter_mass);

		local_count_forward += intersect(
			location, normal,
			forward_box,
			emitter_radius,
			emitter_spin_a, emitter_mass);


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
	//const real_type receiver_distance_plus,
	const real_type receiver_radius,
	const real_type epsilon,
	const real_type emitter_spin_a,
	const real_type emitter_mass,
	const real_type receiver_theta,
	const real_type receiver_phi)
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

	unsigned start_time = (unsigned)time(0);

	for (unsigned int t = 0; t < num_threads; t++)
	{
		long long unsigned int thread_iterations = iterations_per_thread;
		if (t < remainder) thread_iterations++; // Distribute remainder

		long long unsigned int thread_end = current_start + thread_iterations;

		// Each thread gets a different seed based on thread index
		unsigned int thread_seed = t + start_time;

		threads.emplace_back(
			worker_thread,
			current_start,
			thread_end,
			thread_seed,
			emitter_radius,
			receiver_distance,
			//receiver_distance_plus,
			receiver_radius,
			epsilon,
			emitter_spin_a,
			emitter_mass,
			receiver_theta,
			receiver_phi,
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
	const real_type spin = 0.0; // a*

	// --- derived ---
	const real_type spin_root = sqrt(1.0 - spin * spin);

	const real_type emitter_mass_geometrized =
		sqrt(n_geometrized * log(2.0) / (2.0 * pi * (1.0 + spin_root)));

	const real_type emitter_a_geometrized =
		sqrt(n_geometrized * log(2.0) * (1.0 - spin_root) / (2.0 * pi));

	const real_type emitter_r_plus_geometrized =
		sqrt(n_geometrized * log(2.0) * (1.0 + spin_root) / (2.0 * pi));

	const real_type emitter_area_geometrized =
		4.0 * n_geometrized * log(2.0);



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

	// Receiver box angular position (radians); the radius is receiver_distance.
	// theta: polar angle from +z; phi: equatorial angle from +x toward +y.
	// The box is centred at (receiver_distance, theta, phi) and oriented along
	// the local spherical axes there.
	// theta = pi/2, phi = 0 places it at (receiver_distance, 0, 0), matching
	// the original axis-aligned box.
	const real_type receiver_theta = pi / 2.0;
	const real_type receiver_phi = 0.0;

	for (size_t i = 0; i < pos_res; i++)
	{
		cout << "\n=== Step " << (i + 1) << " of " << pos_res << " ===" << endl;

		const real_type receiver_distance_geometrized =
			start_pos + i * pos_step_size;

		//const real_type receiver_distance_plus_geometrized =
		//	receiver_distance_geometrized + epsilon;

		segment_length = receiver_radius_geometrized / 10.0;// epsilon / 10.0;

		// beta function
		const pair<real_type, real_type> collision_count_plus_minus_collision_count =
			get_intersecting_line_density(
				static_cast<long long unsigned int>(n_geometrized),
				emitter_r_plus_geometrized,
				receiver_distance_geometrized,
				//receiver_distance_plus_geometrized,
				receiver_radius_geometrized,
				epsilon,
				emitter_a_geometrized,
				emitter_mass_geometrized,
				receiver_theta,
				receiver_phi);

		// alpha variable
		const real_type gradient_integer =
			collision_count_plus_minus_collision_count.first
			/ epsilon;

		const real_type gradient_integer_sideways =
			collision_count_plus_minus_collision_count.second
			/ (epsilon);

		// g variable
		real_type gradient_strength =
			-gradient_integer
			/
			(2.0 * receiver_radius_geometrized
				* receiver_radius_geometrized
				* receiver_radius_geometrized);

		real_type gradient_strength_sideways =
			-gradient_integer_sideways
			/
			(2.0 * receiver_radius_geometrized
				* receiver_radius_geometrized
				* receiver_radius_geometrized);

		const real_type a_flat_geometrized =
			gradient_strength * receiver_distance_geometrized * log(2)
			/ (8.0 * emitter_mass_geometrized);

		const real_type a_flat_geometrized_sideways =
			gradient_strength_sideways * receiver_distance_geometrized * log(2)
			/ (8.0 * emitter_mass_geometrized);



		const real_type a_Newton_geometrized =
			sqrt(
				n_geometrized * log(2.0)
				/
				(4.0 * pi *
					pow(receiver_distance_geometrized, 4.0))
			);

		const real_type dt_Schwarzschild = sqrt(1 - emitter_r_plus_geometrized / receiver_distance_geometrized);

		const real_type a_Schwarzschild_geometrized =
			emitter_r_plus_geometrized / (pi * pow(receiver_distance_geometrized, 2.0) * dt_Schwarzschild);



		cout << "a_Schwarzschild_geometrized " << a_Schwarzschild_geometrized << endl;
		cout << "a_Newton_geometrized " << a_Newton_geometrized << endl;
		cout << "a_flat_geometrized " << a_flat_geometrized << endl;
		cout << "a_flat_geometrized_sideways " << a_flat_geometrized_sideways << endl;

		cout << a_Schwarzschild_geometrized / a_flat_geometrized << endl;
		cout << endl;

		cout << a_Schwarzschild_geometrized / a_flat_geometrized_sideways << endl;
		cout << endl;

		cout << a_Schwarzschild_geometrized / a_flat_geometrized + a_Schwarzschild_geometrized / a_flat_geometrized_sideways << endl;
		cout << endl;

		cout << a_Newton_geometrized / a_flat_geometrized << endl;
		cout << endl << endl;


		outfile_numerical << receiver_distance_geometrized << " " << a_flat_geometrized << endl;
		outfile_analytical << receiver_distance_geometrized << " " << a_Schwarzschild_geometrized << endl;
		outfile_Newton << receiver_distance_geometrized << " " << a_Newton_geometrized << endl;


	}


}