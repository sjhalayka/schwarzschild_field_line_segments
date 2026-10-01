#include "main.h"

// OpenGL 4 / GLEW / freeglut. GLEW must be included before any other GL header.
#include "GL/glew.h"
#include "GL/freeglut.h"
#pragma comment(lib, "freeglut")
#pragma comment(lib, "glew32")
#include <mutex>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <limits>
#include <cstdint>





const real_type spin = 0.999; // a_*
const real_type theta = pi / 2.0;
const real_type n_geometrized = 1e9; // field line count





// Atomic counter for progress tracking
std::atomic<long long unsigned int> global_progress(0);



// Length of each line segment the ray is broken into
real_type segment_length = 1.0;



// ---------------------------------------------------------------------------
// Oriented Bounding Box
//
// Conventions (physics / ISO 80000-2):
//   theta = polar angle measured from +z, in [0, pi]
//   phi   = azimuthal angle measured from +x toward +y, in (-pi, pi]
//
// Location:    the box centre is stored as spherical (r, theta, phi).
//
// Orientation: the 3x3 orientation matrix is stored as spherical angles
//              (orient_theta, orient_phi, orient_roll). Its columns are the
//              local spherical basis evaluated at (orient_theta, orient_phi):
//
//                  axis[0] = e_r
//                  axis[1] =  cos(roll) e_theta + sin(roll) e_phi
//                  axis[2] = -sin(roll) e_theta + cos(roll) e_phi
//
//              which is always a right-handed orthonormal frame
//              (e_r x e_theta = e_phi). Setting the orientation angles equal
//              to the location angles gives a radially aligned box.
//
// half_extents are measured along axis[0], axis[1], axis[2] respectively.
//
// The Cartesian centre, the axes and the bounding radius are derived data,
// cached by update() so the hot loop does no trigonometry.
// ---------------------------------------------------------------------------
struct OBB
{
	// Stored (spherical) parameters
	real_type r = 0, theta = 0, phi = 0;
	real_type orient_theta = 0, orient_phi = 0, orient_roll = 0;
	vector_3 half_extents;

	// Cached (derived) data
	vector_3 center;
	vector_3 axis[3];
	real_type bounding_radius = 0; // max distance from the origin of any point in the box

	void update()
	{
		// Centre: spherical -> Cartesian
		const real_type st = sin(theta), ct = cos(theta);
		const real_type sp = sin(phi), cp = cos(phi);
		center = vector_3(r * st * cp, r * st * sp, r * ct);

		// Orientation matrix from the spherical basis at (orient_theta, orient_phi)
		const real_type sot = sin(orient_theta), cot = cos(orient_theta);
		const real_type sop = sin(orient_phi), cop = cos(orient_phi);

		vector_3 e_r(sot * cop, sot * sop, cot);
		vector_3 e_theta(cot * cop, cot * sop, -sot);
		vector_3 e_phi(-sop, cop, 0.0);

		const real_type sr = sin(orient_roll), cr = cos(orient_roll);

		axis[0] = e_r;
		axis[1] = e_theta * cr + e_phi * sr;
		axis[2] = e_theta * (-sr) + e_phi * cr;

		bounding_radius = r + vector_3(half_extents).length();
	}
};

inline real_type dot3(const vector_3& a, const vector_3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cartesian -> spherical (r, theta, phi)
void cartesian_to_spherical(const vector_3& p, real_type& r, real_type& theta, real_type& phi)
{
	r = sqrt(dot3(p, p));

	if (r == 0)
	{
		theta = 0;
		phi = 0;
		return;
	}

	real_type c = p.z / r;
	if (c > 1.0) c = 1.0;
	if (c < -1.0) c = -1.0;

	theta = acos(c);
	phi = atan2(p.y, p.x);
}

// Build a box whose orientation follows the local spherical basis at its own
// location (axis[0] points radially outward from the emitter).
OBB make_radial_OBB(
	const real_type r,
	const real_type theta,
	const real_type phi,
	const vector_3 half_extents,
	const real_type roll = 0.0)
{
	OBB box;
	box.r = r;
	box.theta = theta;
	box.phi = phi;
	box.orient_theta = theta;
	box.orient_phi = phi;
	box.orient_roll = roll;
	box.half_extents = half_extents;
	box.update();
	return box;
}

// Rigidly translate a box by a world-space offset. The orientation matrix is
// kept fixed; only the spherical location is recomputed.
OBB translate_OBB(OBB box, vector_3 offset)
{
	OBB out = box;
	cartesian_to_spherical(box.center + offset, out.r, out.theta, out.phi);
	out.update();
	return out;
}



// Builds the receiver box and the two boxes shifted by epsilon from it.
// Used by the worker threads and by the visualisation, so both always agree.
void make_receiver_boxes(
	const real_type receiver_distance,
	const real_type receiver_radius,
	const real_type epsilon,
	OBB& receiver_box,
	OBB& right_box,
	OBB& forward_box)
{
	// Receiver sits on the +x axis: r = receiver_distance, theta = pi/2, phi = 0.
	// Its radial orientation there is axis[0] = +x, axis[1] = -z, axis[2] = +y.
	receiver_box = make_radial_OBB(
		receiver_distance,
		0.5 * pi,
		0.0,
		vector_3(receiver_radius, receiver_radius, receiver_radius));

	// Same orientation, shifted by epsilon along world +x and world +z
	right_box = translate_OBB(receiver_box, vector_3(epsilon, 0.0, 0.0));
	//	forward_box = translate_OBB(receiver_box, vector_3(0.0, 0.0, epsilon));
	forward_box = make_radial_OBB(
		receiver_distance,
		theta,
		epsilon / receiver_distance,
		vector_3(receiver_radius, receiver_radius, receiver_radius));
}



// ---------------------------------------------------------------------------
// State shared between the simulation thread and the OpenGL (GLUT) thread
// ---------------------------------------------------------------------------
struct render_snapshot
{
	OBB receiver_box;
	OBB forward_box;
	real_type emitter_radius = 0;
	size_t step = 0;
	size_t step_count = 0;
	bool valid = false;
	bool finished = false;
};

std::mutex render_mutex;
render_snapshot shared_snapshot;

// Set when the window is closed, so the simulation can stop early
std::atomic<bool> abort_requested(false);

void publish_render_state(
	const OBB& receiver_box,
	const OBB& forward_box,
	const real_type emitter_radius,
	const size_t step,
	const size_t step_count)
{
	std::lock_guard<std::mutex> lock(render_mutex);
	shared_snapshot.receiver_box = receiver_box;
	shared_snapshot.forward_box = forward_box;
	shared_snapshot.emitter_radius = emitter_radius;
	shared_snapshot.step = step;
	shared_snapshot.step_count = step_count;
	shared_snapshot.valid = true;
}

void publish_simulation_finished()
{
	std::lock_guard<std::mutex> lock(render_mutex);
	shared_snapshot.finished = true;
}



// ---------------------------------------------------------------------------
// Field line capture for the visualisation
//
// intersect_OBB records the first max_recorded_field_lines field lines it
// traces (from the ray origin on the emitter surface to where the march
// stopped). The set is cleared at the start of every
// get_intersecting_line_density call, so it always holds the lines of the
// current simulation step. Only calls made with record_field_line = true are captured, so a
// field line that is traced against several boxes is stored only once.
// ---------------------------------------------------------------------------
const long long unsigned int max_recorded_field_lines
= static_cast<long long unsigned int>(1e6);







struct field_line
{
	vector_3 start, end;
	bool is_first_segment; // true only for the segment that begins on the emitter surface
};

std::mutex field_line_mutex;
vector<field_line> recorded_field_lines;             // guarded by field_line_mutex
std::atomic<size_t> recorded_field_line_count(0);    // lock-free early-out for the hot loop
size_t recorded_field_line_generation = 0;           // bumped on every clear; guarded by field_line_mutex

// Discards the previously recorded field lines so the next batch starts empty
void clear_recorded_field_lines()
{
	std::lock_guard<std::mutex> lock(field_line_mutex);
	recorded_field_lines.clear();
	recorded_field_line_count.store(0, std::memory_order_relaxed);
	recorded_field_line_generation++;
}

void record_field_line(const vector_3& start, const vector_3& end, const bool is_first_segment)
{
	// Cheap check first, so the worker threads never touch the mutex once full
	if (recorded_field_line_count.load(std::memory_order_relaxed) >= max_recorded_field_lines)
		return;

	std::lock_guard<std::mutex> lock(field_line_mutex);

	if (recorded_field_lines.size() >= max_recorded_field_lines)
		return;

	recorded_field_lines.push_back({ start, end, is_first_segment });
	recorded_field_line_count.store(recorded_field_lines.size(), std::memory_order_relaxed);
}



// Point-in-box test for a finite line segment against an oriented bounding box.
// Returns true when the segment's midpoint lies inside the box.
bool intersect_segment_OBB(
	const OBB& box,
	const vector_3 segment_start,
	const vector_3 segment_end)
{
	vector_3 segment_mid_point;
	segment_mid_point.x = (segment_start.x + segment_end.x) * 0.5;
	segment_mid_point.y = (segment_start.y + segment_end.y) * 0.5;
	segment_mid_point.z = (segment_start.z + segment_end.z) * 0.5;

	// Project the midpoint (relative to the centre) onto the box's local axes
	const vector_3 d = segment_mid_point - box.center;

	return
		fabs(dot3(d, box.axis[0])) <= box.half_extents.x &&
		fabs(dot3(d, box.axis[1])) <= box.half_extents.y &&
		fabs(dot3(d, box.axis[2])) <= box.half_extents.z;
}

real_type intersect_OBB(
	const OBB& box,
	vector_3 ray_origin, vector_3 ray_dir,
	const bool record_this_field_line = false)
{
	// No point of the box is farther from the origin than bounding_radius;
	// one extra segment of margin covers segments that straddle that sphere.
	const real_type max_distance = box.bounding_radius + segment_length;

	vector_3 ray = ray_dir * segment_length;

	vector_3 segment_start = ray_origin;
	vector_3 segment_end = ray_origin + ray;

	if (record_this_field_line)
		record_field_line(segment_start, segment_end, true);

	real_type total_length = 0;

	while (segment_end.length() < max_distance)
	{
		if (intersect_segment_OBB(box, segment_start, segment_end))
			total_length += (segment_end - segment_start).length();

		segment_start = segment_end;
		segment_end += ray;

		if (record_this_field_line)
			record_field_line(segment_start, segment_end, false);
	}


	return total_length;
}

real_type intersect(
	const vector_3 location,
	const vector_3 normal,
	const OBB& box,
	const bool record_this_field_line = false)
{
	return intersect_OBB(box, location, normal, record_this_field_line);
}

// Thread-local versions of random functions that take generator and distribution as parameters
vector_3 random_cosine_weighted_hemisphere(vector_3 normal,
	std::mt19937& local_gen, std::uniform_real_distribution<real_type>& local_dis)
{
	vector_3 r = vector_3(local_dis(local_gen), local_dis(local_gen), 0.0);
	vector_3 uu = normal.cross(vector_3(0.0, 1.0, 1.0)).normalize();
	vector_3 vv = uu.cross(normal);

	real_type ra = sqrt(r.y);
	real_type rx = ra * cos(2.0 * pi * r.x);
	real_type ry = ra * sin(2.0 * pi * r.x);
	real_type rz = sqrt(1.0 - r.y);
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


// ---------------------------------------------------------------------------
// Low-discrepancy (quasi-random) sampling: Owen-scrambled 4D Sobol sequence
//
// Each iteration needs one sample of the 4D product space S2 x S2 (location
// and r). A plain Kronecker / R_d sequence is a rank-1 lattice, so its 2D
// projections show up on the sphere as visible rows. Sobol points with a
// nested uniform (Owen) scramble keep the low discrepancy and the 4D
// stratification, but the lattice structure is destroyed, so the points look
// evenly spread with no rows. Sample i depends only on i, so the thread split
// does not matter. Best balance when n is a power of 2, but any n works.
// ---------------------------------------------------------------------------
inline uint32_t reverse_bits_32(uint32_t x)
{
	x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
	x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
	x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
	x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
	return (x >> 16) | (x << 16);
}

struct sobol_4d_directions
{
	uint32_t v[4][32];

	sobol_4d_directions()
	{
		// Joe & Kuo parameters for dimensions 2..4: degree s, coefficients a, initial m
		const int      s[3] = { 1, 2, 3 };
		const uint32_t a[3] = { 0, 1, 1 };
		const uint32_t m[3][3] = { { 1, 0, 0 }, { 1, 3, 0 }, { 1, 3, 1 } };

		for (int k = 0; k < 32; k++)
			v[0][k] = 1u << (31 - k); // dimension 1: van der Corput

		for (int d = 1; d < 4; d++)
		{
			const int deg = s[d - 1];

			for (int k = 0; k < deg; k++)
				v[d][k] = m[d - 1][k] << (31 - k);

			for (int k = deg; k < 32; k++)
			{
				uint32_t x = v[d][k - deg] ^ (v[d][k - deg] >> deg);

				for (int j = 1; j < deg; j++)
					if ((a[d - 1] >> (deg - 1 - j)) & 1u)
						x ^= v[d][k - j];

				v[d][k] = x;
			}
		}
	}
};

const sobol_4d_directions sobol_directions;

inline uint32_t sobol_32(uint32_t index, const int dim)
{
	uint32_t x = 0;
	for (int k = 0; index != 0; index >>= 1, k++)
		if (index & 1u)
			x ^= sobol_directions.v[dim][k];
	return x;
}

inline uint32_t hash_32(uint32_t x)
{
	x ^= x >> 16; x *= 0x21f0aaadu;
	x ^= x >> 15; x *= 0xd35a2d97u;
	x ^= x >> 15;
	return x;
}

// Laine-Karras style nested uniform (Owen) scramble
inline uint32_t owen_scramble_32(uint32_t x, const uint32_t seed)
{
	x = reverse_bits_32(x);
	x ^= x * 0x3d20adeau;
	x += seed;
	x *= (seed >> 16) | 1u;
	x ^= x * 0x05526c56u;
	x ^= x * 0x53a22864u;
	return reverse_bits_32(x);
}

const uint32_t sobol_scramble_seed = 0x9e3779b9u;

void qmc_sample_4d(const long long unsigned int i, real_type u[4])
{
	for (int d = 0; d < 4; d++)
	{
		const uint32_t seed = hash_32(sobol_scramble_seed + 0x632be5abu * static_cast<uint32_t>(d + 1));
		const uint32_t bits = owen_scramble_32(sobol_32(static_cast<uint32_t>(i), d), seed);
		u[d] = (static_cast<real_type>(bits) + 0.5) / 4294967296.0;
	}
}

// Area-preserving map from the unit square to the unit sphere (Archimedes'
// hat-box theorem): z uniform in [-1, 1], azimuth uniform in [0, 2 pi).
// This is the same map random_unit_vector uses, so an evenly spread square
// becomes an evenly spread sphere.
vector_3 unit_square_to_sphere(const real_type u, const real_type v)
{
	const real_type z = 2.0 * u - 1.0;
	const real_type a = 2.0 * pi * v;
	const real_type s = sqrt(fmax(0.0, 1.0 - z * z));

	return vector_3(s * cos(a), s * sin(a), z);
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


	// The receiver boxes do not depend on the iteration, so build them once.
	OBB receiver_box, right_box, forward_box;
	make_receiver_boxes(receiver_distance, receiver_radius, epsilon, receiver_box, right_box, forward_box);

	// Update progress every N iterations to reduce atomic overhead
	const long long unsigned int progress_update_interval = 10000;
	long long unsigned int local_progress = 0;

	for (long long unsigned int i = start_idx; i < end_idx; i++)
	{
		// One low-discrepancy sample of S2 x S2 per field line
		//real_type u[4];
		//qmc_sample_4d(i, u);

		//vector_3 location = unit_square_to_sphere(u[0], u[1]);

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
		//vector_3 r = unit_square_to_sphere(u[2], u[3]);
		vector_3 r = random_unit_vector(local_gen, local_dis);


		r.x *= emitter_radius;
		r.y *= emitter_radius;
		r.z *= emitter_radius;

		vector_3 normal = (location - r).normalize();



		// The same field line is traced against all three boxes; record it once
		local_count += intersect(location, normal, receiver_box, true);
		local_count_plus += intersect(location, normal, right_box);
		local_count_forward += intersect(location, normal, forward_box);



		// Update global progress periodically
		local_progress++;
		if (local_progress >= progress_update_interval)
		{
			global_progress.fetch_add(local_progress, std::memory_order_relaxed);
			local_progress = 0;

			// Stop early if the window was closed
			if (abort_requested.load(std::memory_order_relaxed))
				break;
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
		real_type progress = static_cast<real_type>(current) / static_cast<real_type>(total_iterations);

		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

		// Estimate time remaining
		real_type eta_seconds = 0;
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

	// Start a fresh set of recorded field lines for this call
	// (done before any worker thread is launched)
	clear_recorded_field_lines();

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

	unsigned curr_time = (unsigned)time(0);

	for (unsigned int t = 0; t < num_threads; t++)
	{
		long long unsigned int thread_iterations = iterations_per_thread;
		if (t < remainder) thread_iterations++; // Distribute remainder

		long long unsigned int thread_end = current_start + thread_iterations;

		// Each thread gets a different seed based on thread index
		unsigned int thread_seed = t + curr_time;

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











real_type expr(real_type r, real_type a, real_type M, real_type t)
{
	const real_type a2 = a * a;
	const real_type a4 = a2 * a2;
	const real_type r2 = r * r;
	const real_type r3 = r2 * r;
	const real_type s = std::sin(t);
	const real_type c = std::cos(t);
	const real_type s2 = s * s;
	const real_type c2 = c * c;

	const real_type sum = a2 + r2;              // a^2 + r^2
	const real_type delta = a2 + r * (r - 2.0 * M); // a^2 + r(r - 2M)

	const real_type num =
		r * (sum * (a4 + a2 * r * (r - 3.0 * M) + M * r3)
			- a2 * s2 * delta * delta)
		- a2 * sum * c2 * (a2 * (M + r) + r2 * (r - 3.0 * M));

	const real_type D = sum * sum - s2 * (a4 + a2 * r * (r - 2.0 * M));

	const real_type den = D * D * std::sqrt(delta * (a2 * c2 + r2) / D);

	return num / den;
}



double deta_dr(double r, double theta, double M, double a) {
	const double s2 = std::sin(theta) * std::sin(theta), c2 = 1.0 - s2;
	const double r2a2 = r * r + a * a;
	const double Sigma = r * r + a * a * c2;
	const double Delta = r * r - 2.0 * M * r + a * a;
	const double A = r2a2 * r2a2 - a * a * Delta * s2;
	const double dA_half = 2.0 * r * r2a2 - a * a * (r - M) * s2;   // (dA/dr)/2
	const double num = 2.0 * A * (r * Delta + (r - M) * Sigma) - Sigma * Delta * dA_half * 2.0 / 2.0 * 1.0;
	return num / (2.0 * std::pow(A, 1.5) * std::sqrt(Sigma * Delta));
}


void run_simulation()
{
	ofstream outfile_numerical("Schwarzschild_numerical");
	ofstream outfile_analytical("Schwarzschild_analytical");
	ofstream outfile_Newton("Newton_analytical");








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

	start_pos *= 2.0;

	real_type end_pos = start_pos;// start_pos * 2.0;

	const size_t pos_res = 2; // Minimum 2 steps

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

		// Hand the boxes for this step to the renderer
		{
			OBB receiver_box, right_box, forward_box;

			make_receiver_boxes(
				receiver_distance_geometrized,
				receiver_radius_geometrized,
				epsilon,
				receiver_box, right_box, forward_box);

			publish_render_state(
				receiver_box, forward_box,
				emitter_r_plus_geometrized,
				i + 1, pos_res);
		}

		// beta function
		const pair<real_type, real_type> collision_count_plus_minus_collision_count =
			get_intersecting_line_density(
				static_cast<long long unsigned int>(n_geometrized),
				emitter_r_plus_geometrized,
				receiver_distance_geometrized,
				receiver_distance_plus_geometrized,
				receiver_radius_geometrized,
				epsilon);

		if (abort_requested.load(std::memory_order_relaxed))
		{
			cout << "Simulation aborted." << endl;
			break;
		}

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

		const real_type a_flat_geometrized =
			gradient_strength * receiver_distance_geometrized * log(2)
			/ (8.0 * emitter_mass_geometrized);





		const real_type gradient_integer_forward =
			collision_count_plus_minus_collision_count.second
			/ epsilon;

		// g variable
		real_type gradient_strength_forward =
			-gradient_integer_forward
			/
			(2.0 * receiver_radius_geometrized
				* receiver_radius_geometrized
				* receiver_radius_geometrized);

		const real_type a_flat_geometrized_forward =
			gradient_strength_forward * receiver_distance_geometrized * log(2)
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



		const real_type sigma = receiver_distance_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized * cos(theta) * cos(theta);
		const real_type delta = receiver_distance_geometrized * receiver_distance_geometrized - 2 * emitter_mass_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized;
		const real_type A = (receiver_distance_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized) * (receiver_distance_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized) - emitter_a_geometrized * emitter_a_geometrized * (receiver_distance_geometrized * receiver_distance_geometrized - 2 * emitter_mass_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized) * sin(theta) * sin(theta);

		//const real_type dt_Kerr = sqrt(sigma*delta/A);
		//const real_type dt_Kerr = sqrt((receiver_distance_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized * cos(theta) * cos(theta)) * (receiver_distance_geometrized * receiver_distance_geometrized - 2 * emitter_mass_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized) / ((receiver_distance_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized) * (receiver_distance_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized) - emitter_a_geometrized * emitter_a_geometrized * (receiver_distance_geometrized * receiver_distance_geometrized - 2 * emitter_mass_geometrized * receiver_distance_geometrized + emitter_a_geometrized * emitter_a_geometrized) * sin(theta) * sin(theta)));



//		const real_type a_Kerr_geometrized =

//			(2.0/pi)*expr(receiver_distance_geometrized, emitter_a_geometrized, emitter_mass_geometrized, theta);


		double a_Kerr_geometrized = (2.0 / pi) * deta_dr(receiver_distance_geometrized, theta, emitter_mass_geometrized, emitter_a_geometrized);

		//cout << a_Kerr_geometrized << " " << a_Kerr_geometrized_ << endl;
		//exit(0);

			//emitter_r_plus_geometrized / (pi * pow(receiver_distance_geometrized, 2.0) * dt_Schwarzschild);






		cout << "a_Kerr_geometrized " << a_Kerr_geometrized << endl;

		//		cout << "a_Schwarzschild_geometrized " << a_Schwarzschild_geometrized << endl;
		cout << "a_Newton_geometrized " << a_Newton_geometrized << endl;
		cout << "a_flat_geometrized " << a_flat_geometrized << endl;
		cout << "a_flat_geometrized_forward " << a_flat_geometrized_forward << endl;


		cout << a_Kerr_geometrized / a_flat_geometrized << endl;
		cout << endl;

		//cout << a_Schwarzschild_geometrized / a_flat_geometrized_forward << endl;
		//cout << endl;

		cout << a_Newton_geometrized / a_flat_geometrized << endl;
		cout << endl << endl;


		outfile_numerical << receiver_distance_geometrized << " " << a_flat_geometrized << endl;
		outfile_analytical << receiver_distance_geometrized << " " << a_Schwarzschild_geometrized << endl;
		outfile_Newton << receiver_distance_geometrized << " " << a_Newton_geometrized << endl;


	}



	publish_simulation_finished();
}



// ===========================================================================
// OpenGL 4 visualisation (freeglut + GLEW, core profile)
//
// Controls:
//   left drag    orbit the camera
//   mouse wheel  zoom
//   f            toggle translucent box faces
//   e            toggle emitter (horizon) wireframe
//   a            toggle axes (x red, y green, z blue) at the receiver centre
//   l            toggle field lines (first 1000 traced by intersect_OBB)
//   s            toggle field line start markers (small spheres on the emitter)
//   [ / ]        make the field line cylinders (and markers) thinner / thicker
//   - / =        make the start markers smaller / larger relative to the lines
//   r            reset camera
//   Esc          quit (also stops the simulation)
//
// Everything is drawn relative to the midpoint of the two boxes, computed in
// real_type (long real_type) before converting to float, so the boxes stay
// precise even when they are far from the origin.
// ===========================================================================

struct vec3f
{
	float x, y, z;
};

inline vec3f vec3f_sub(const vec3f& a, const vec3f& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline float vec3f_dot(const vec3f& a, const vec3f& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline vec3f vec3f_cross(const vec3f& a, const vec3f& b)
{
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
inline vec3f vec3f_normalize(const vec3f& a)
{
	const float len = sqrtf(vec3f_dot(a, a));
	return (len > 0.0f) ? vec3f{ a.x / len, a.y / len, a.z / len } : a;
}

// Column-major 4x4 matrix, as OpenGL expects
struct mat4f
{
	float m[16];
};

mat4f mat4f_identity()
{
	mat4f r = {};
	r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
	return r;
}

mat4f mat4f_multiply(const mat4f& a, const mat4f& b)
{
	mat4f r = {};

	for (int col = 0; col < 4; col++)
		for (int row = 0; row < 4; row++)
			for (int k = 0; k < 4; k++)
				r.m[col * 4 + row] += a.m[k * 4 + row] * b.m[col * 4 + k];

	return r;
}

mat4f mat4f_translate(const vec3f& t)
{
	mat4f r = mat4f_identity();
	r.m[12] = t.x;
	r.m[13] = t.y;
	r.m[14] = t.z;
	return r;
}

mat4f mat4f_perspective(const float fovy_radians, const float aspect, const float z_near, const float z_far)
{
	const float f = 1.0f / tanf(0.5f * fovy_radians);

	mat4f r = {};
	r.m[0] = f / aspect;
	r.m[5] = f;
	r.m[10] = (z_far + z_near) / (z_near - z_far);
	r.m[11] = -1.0f;
	r.m[14] = 2.0f * z_far * z_near / (z_near - z_far);
	return r;
}

mat4f mat4f_look_at(const vec3f& eye, const vec3f& center, const vec3f& up)
{
	const vec3f f = vec3f_normalize(vec3f_sub(center, eye));
	const vec3f s = vec3f_normalize(vec3f_cross(f, up));
	const vec3f u = vec3f_cross(s, f);

	mat4f r = mat4f_identity();
	r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;
	r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;
	r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
	r.m[12] = -vec3f_dot(s, eye);
	r.m[13] = -vec3f_dot(u, eye);
	r.m[14] = vec3f_dot(f, eye);
	return r;
}



// --- GL objects ---
GLuint viz_program = 0;
GLint viz_mvp_location = -1;
GLint viz_color_location = -1;

GLuint viz_dynamic_vao = 0, viz_dynamic_vbo = 0; // re-uploaded every frame (boxes, axes)
GLuint viz_sphere_vao = 0, viz_sphere_vbo = 0;   // built once (emitter wireframe)
GLsizei viz_sphere_vertex_count = 0;

// Field line cylinders: separate lit program, interleaved position + normal
GLuint viz_lit_program = 0;
GLint viz_lit_mvp_location = -1;
GLint viz_lit_color_location = -1;
GLint viz_lit_light_dir_location = -1;

GLuint viz_cylinder_vao = 0, viz_cylinder_vbo = 0;
GLsizei viz_cylinder_vertex_count = 0;

// What the cylinder buffer was last built from (rebuilt when any of these change)
size_t viz_cylinder_built_count = 0;
size_t viz_cylinder_built_generation = static_cast<size_t>(-1);
vector_3 viz_cylinder_built_origin;
float viz_cylinder_built_radius = -1.0f;

const int viz_cylinder_sides = 8;
float viz_field_line_radius_scale = 0.02f; // cylinder radius as a fraction of the receiver box size

// Field line start markers: one low-poly unit sphere mesh, drawn instanced
// (one instance per recorded field line) at each line's start point on the
// emitter surface. Shares the lit fragment shader with the cylinders.
GLuint viz_marker_program = 0;
GLint viz_marker_mvp_location = -1;
GLint viz_marker_radius_location = -1;
GLint viz_marker_color_location = -1;
GLint viz_marker_light_dir_location = -1;

GLuint viz_marker_vao = 0, viz_marker_mesh_vbo = 0, viz_marker_instance_vbo = 0;
GLsizei viz_marker_mesh_vertex_count = 0;
GLsizei viz_marker_instance_count = 0;

const int viz_marker_stacks = 8;
const int viz_marker_slices = 12;
float viz_marker_radius_factor = 15.0f; // marker radius as a multiple of the cylinder radius

// --- Window / camera / UI state ---
int viz_window_width = 1280;
int viz_window_height = 720;

const float viz_default_yaw = 1.1f;
const float viz_default_pitch = 0.35f;

float viz_camera_yaw = viz_default_yaw;
float viz_camera_pitch = viz_default_pitch;
float viz_camera_distance = 1.0f;
float viz_default_distance = 1.0f;
bool viz_camera_initialized = false;

bool viz_dragging = false;
int viz_last_mouse_x = 0, viz_last_mouse_y = 0;

bool viz_show_faces = true;
bool viz_show_emitter = true;
bool viz_show_axes = true;
bool viz_show_field_lines = true;
bool viz_show_markers = true;

std::string viz_last_title;



const char* viz_vertex_shader_source = R"(
#version 400 core
layout(location = 0) in vec3 position;
uniform mat4 mvp;
void main()
{
	gl_Position = mvp * vec4(position, 1.0);
}
)";

const char* viz_fragment_shader_source = R"(
#version 400 core
uniform vec4 color;
out vec4 frag_color;
void main()
{
	frag_color = color;
}
)";

// Headlight-shaded cylinders; abs() makes the shading two-sided
const char* viz_lit_vertex_shader_source = R"(
#version 400 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
uniform mat4 mvp;
out vec3 v_normal;
void main()
{
	v_normal = normal;
	gl_Position = mvp * vec4(position, 1.0);
}
)";

const char* viz_lit_fragment_shader_source = R"(
#version 400 core
uniform vec4 color;
uniform vec3 light_dir;
in vec3 v_normal;
out vec4 frag_color;
void main()
{
	float diffuse = abs(dot(normalize(v_normal), light_dir));
	frag_color = vec4(color.rgb * (0.3 + 0.7 * diffuse), color.a);
}
)";

// Instanced unit sphere: the mesh vertex real_types as its own normal
const char* viz_marker_vertex_shader_source = R"(
#version 400 core
layout(location = 0) in vec3 position;
layout(location = 2) in vec3 instance_offset;
uniform mat4 mvp;
uniform float radius;
out vec3 v_normal;
void main()
{
	v_normal = position;
	gl_Position = mvp * vec4(instance_offset + radius * position, 1.0);
}
)";

GLuint viz_compile_shader(const GLenum type, const char* source)
{
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint ok = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);

	if (!ok)
	{
		char log[2048];
		glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
		cout << "Shader compile error:\n" << log << endl;
		glDeleteShader(shader);
		return 0;
	}

	return shader;
}

GLuint viz_link_program(const char* vertex_source, const char* fragment_source)
{
	const GLuint vs = viz_compile_shader(GL_VERTEX_SHADER, vertex_source);
	const GLuint fs = viz_compile_shader(GL_FRAGMENT_SHADER, fragment_source);

	if (vs == 0 || fs == 0)
		return 0;

	GLuint program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glDeleteShader(vs);
	glDeleteShader(fs);

	GLint ok = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &ok);

	if (!ok)
	{
		char log[2048];
		glGetProgramInfoLog(program, sizeof(log), nullptr, log);
		cout << "Program link error:\n" << log << endl;
		glDeleteProgram(program);
		return 0;
	}

	return program;
}

// Latitude / longitude triangulated unit sphere, positions only (GL_TRIANGLES)
void viz_build_marker_mesh()
{
	vector<float> v;

	auto push_unit = [&](const real_type theta, const real_type phi)
		{
			v.push_back(static_cast<float>(sin(theta) * cos(phi)));
			v.push_back(static_cast<float>(sin(theta) * sin(phi)));
			v.push_back(static_cast<float>(cos(theta)));
		};

	for (int i = 0; i < viz_marker_stacks; i++)
	{
		const real_type t0 = pi * i / viz_marker_stacks;
		const real_type t1 = pi * (i + 1) / viz_marker_stacks;

		for (int j = 0; j < viz_marker_slices; j++)
		{
			const real_type p0 = 2.0 * pi * j / viz_marker_slices;
			const real_type p1 = 2.0 * pi * (j + 1) / viz_marker_slices;

			push_unit(t0, p0); push_unit(t1, p0); push_unit(t1, p1);
			push_unit(t0, p0); push_unit(t1, p1); push_unit(t0, p1);
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, viz_marker_mesh_vbo);
	glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STATIC_DRAW);
	viz_marker_mesh_vertex_count = static_cast<GLsizei>(v.size() / 3);
}

bool viz_init_gl()
{
	viz_program = viz_link_program(viz_vertex_shader_source, viz_fragment_shader_source);
	viz_lit_program = viz_link_program(viz_lit_vertex_shader_source, viz_lit_fragment_shader_source);
	viz_marker_program = viz_link_program(viz_marker_vertex_shader_source, viz_lit_fragment_shader_source);

	if (viz_program == 0 || viz_lit_program == 0 || viz_marker_program == 0)
		return false;

	viz_marker_mvp_location = glGetUniformLocation(viz_marker_program, "mvp");
	viz_marker_radius_location = glGetUniformLocation(viz_marker_program, "radius");
	viz_marker_color_location = glGetUniformLocation(viz_marker_program, "color");
	viz_marker_light_dir_location = glGetUniformLocation(viz_marker_program, "light_dir");

	viz_lit_mvp_location = glGetUniformLocation(viz_lit_program, "mvp");
	viz_lit_color_location = glGetUniformLocation(viz_lit_program, "color");
	viz_lit_light_dir_location = glGetUniformLocation(viz_lit_program, "light_dir");

	viz_mvp_location = glGetUniformLocation(viz_program, "mvp");
	viz_color_location = glGetUniformLocation(viz_program, "color");

	// Dynamic buffer
	glGenVertexArrays(1, &viz_dynamic_vao);
	glGenBuffers(1, &viz_dynamic_vbo);
	glBindVertexArray(viz_dynamic_vao);
	glBindBuffer(GL_ARRAY_BUFFER, viz_dynamic_vbo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

	// Static sphere buffer (filled once the emitter radius is known)
	glGenVertexArrays(1, &viz_sphere_vao);
	glGenBuffers(1, &viz_sphere_vbo);
	glBindVertexArray(viz_sphere_vao);
	glBindBuffer(GL_ARRAY_BUFFER, viz_sphere_vbo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

	// Field line cylinder buffer: position (3) + normal (3), interleaved
	glGenVertexArrays(1, &viz_cylinder_vao);
	glGenBuffers(1, &viz_cylinder_vbo);
	glBindVertexArray(viz_cylinder_vao);
	glBindBuffer(GL_ARRAY_BUFFER, viz_cylinder_vbo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));

	// Start markers: static unit sphere mesh (location 0) + per-instance offset (location 2)
	glGenVertexArrays(1, &viz_marker_vao);
	glGenBuffers(1, &viz_marker_mesh_vbo);
	glGenBuffers(1, &viz_marker_instance_vbo);
	glBindVertexArray(viz_marker_vao);

	glBindBuffer(GL_ARRAY_BUFFER, viz_marker_mesh_vbo);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

	glBindBuffer(GL_ARRAY_BUFFER, viz_marker_instance_vbo);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
	glVertexAttribDivisor(2, 1);

	viz_build_marker_mesh();

	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	cout << "OpenGL " << glGetString(GL_VERSION) << " | " << glGetString(GL_RENDERER) << endl;

	return true;
}

// Latitude / longitude wireframe of the emitter sphere, in world coordinates
void viz_build_emitter_sphere(const real_type radius)
{
	vector<float> v;

	const int grid_degrees = 5;    // spacing between grid lines
	const int segment_degrees = 1; // resolution along each line

	auto push_point = [&](const real_type theta, const real_type phi)
		{
			v.push_back(static_cast<float>(radius * sin(theta) * cos(phi)));
			v.push_back(static_cast<float>(radius * sin(theta) * sin(phi)));
			v.push_back(static_cast<float>(radius * cos(theta)));
		};

	const real_type deg = pi / 180.0;

	// Circles of constant theta
	for (int t = grid_degrees; t < 180; t += grid_degrees)
		for (int p = 0; p < 360; p += segment_degrees)
		{
			push_point(t * deg, p * deg);
			push_point(t * deg, (p + segment_degrees) * deg);
		}

	// Meridians (constant phi)
	for (int p = 0; p < 360; p += grid_degrees)
		for (int t = 0; t < 180; t += segment_degrees)
		{
			push_point(t * deg, p * deg);
			push_point((t + segment_degrees) * deg, p * deg);
		}

	glBindBuffer(GL_ARRAY_BUFFER, viz_sphere_vbo);
	glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STATIC_DRAW);
	viz_sphere_vertex_count = static_cast<GLsizei>(v.size() / 3);
}

// Corner i uses sign bit 0 for axis[0], bit 1 for axis[1], bit 2 for axis[2].
// Output is relative to 'origin'.
void viz_get_OBB_corners(const OBB& box, const vector_3& origin, vector_3 corners[8])
{
	const vector_3 c = box.center - origin;
	vector_3 a0 = box.axis[0], a1 = box.axis[1], a2 = box.axis[2];

	for (int i = 0; i < 8; i++)
	{
		const real_type s0 = (i & 1) ? box.half_extents.x : -box.half_extents.x;
		const real_type s1 = (i & 2) ? box.half_extents.y : -box.half_extents.y;
		const real_type s2 = (i & 4) ? box.half_extents.z : -box.half_extents.z;

		vector_3 p = c;
		p += a0 * s0;
		p += a1 * s1;
		p += a2 * s2;

		corners[i] = p;
	}
}

inline void viz_push(vector<float>& v, const vector_3& p)
{
	v.push_back(static_cast<float>(p.x));
	v.push_back(static_cast<float>(p.y));
	v.push_back(static_cast<float>(p.z));
}

void viz_build_OBB_edges(const OBB& box, const vector_3& origin, vector<float>& v)
{
	vector_3 c[8];
	viz_get_OBB_corners(box, origin, c);

	// 12 edges: join corners that differ in exactly one sign bit
	for (int i = 0; i < 8; i++)
		for (int bit = 1; bit <= 4; bit <<= 1)
			if (!(i & bit))
			{
				viz_push(v, c[i]);
				viz_push(v, c[i | bit]);
			}
}

void viz_build_OBB_faces(const OBB& box, const vector_3& origin, vector<float>& v)
{
	vector_3 c[8];
	viz_get_OBB_corners(box, origin, c);

	static const int quads[6][4] =
	{
		{ 0, 2, 6, 4 }, { 1, 3, 7, 5 }, // -axis0, +axis0
		{ 0, 1, 5, 4 }, { 2, 3, 7, 6 }, // -axis1, +axis1
		{ 0, 1, 3, 2 }, { 4, 5, 7, 6 }  // -axis2, +axis2
	};

	for (int f = 0; f < 6; f++)
	{
		const int* q = quads[f];
		viz_push(v, c[q[0]]); viz_push(v, c[q[1]]); viz_push(v, c[q[2]]);
		viz_push(v, c[q[0]]); viz_push(v, c[q[2]]); viz_push(v, c[q[3]]);
	}
}

// Appends an open cylinder (no end caps) from a to b as GL_TRIANGLES,
// interleaved position + normal. a and b are already relative to the
// render origin, so the float conversion keeps full precision near it.
void viz_append_cylinder(vector<float>& v, const vector_3& a, const vector_3& b, const real_type radius)
{
	vector_3 dir = b - a;
	const real_type len = dir.length();

	if (len <= 0)
		return;

	dir = dir / len;

	// Any vector not parallel to dir, then an orthonormal pair around it
	const vector_3 helper = (fabs(dir.x) < 0.9) ? vector_3(1, 0, 0) : vector_3(0, 1, 0);
	vector_3 u = dir.cross(helper);
	u.normalize();
	vector_3 w = dir.cross(u);
	w.normalize();

	auto push_vertex = [&](const vector_3& base, vector_3 n)
		{
			vector_3 p = n * radius;
			p += base;

			v.push_back(static_cast<float>(p.x));
			v.push_back(static_cast<float>(p.y));
			v.push_back(static_cast<float>(p.z));
			v.push_back(static_cast<float>(n.x));
			v.push_back(static_cast<float>(n.y));
			v.push_back(static_cast<float>(n.z));
		};

	for (int i = 0; i < viz_cylinder_sides; i++)
	{
		const real_type t0 = 2.0 * pi * i / viz_cylinder_sides;
		const real_type t1 = 2.0 * pi * (i + 1) / viz_cylinder_sides;

		vector_3 n0 = u * cos(t0);
		n0 += w * sin(t0);
		vector_3 n1 = u * cos(t1);
		n1 += w * sin(t1);

		push_vertex(a, n0); push_vertex(b, n0); push_vertex(b, n1);
		push_vertex(a, n0); push_vertex(b, n1); push_vertex(a, n1);
	}
}

// Rebuilds the cylinder buffer and the start marker instance buffer if new
// field lines arrived, the render origin moved (new simulation step) or the
// thickness changed.
void viz_update_field_line_buffers(const vector_3& origin, const real_type radius)
{
	vector<field_line> lines;
	{
		std::lock_guard<std::mutex> lock(field_line_mutex);

		// The generation check catches a clear followed by a refill to the
		// same count between two frames
		const bool up_to_date =
			recorded_field_line_generation == viz_cylinder_built_generation &&
			recorded_field_lines.size() == viz_cylinder_built_count &&
			viz_cylinder_built_origin == origin &&
			viz_cylinder_built_radius == static_cast<float>(radius);

		if (up_to_date)
			return;

		lines = recorded_field_lines;
		viz_cylinder_built_generation = recorded_field_line_generation;
	}

	vector<float> v;
	v.reserve(lines.size() * viz_cylinder_sides * 6 * 6);

	for (const field_line& fl : lines)
		viz_append_cylinder(v, fl.start - origin, fl.end - origin, radius);

	glBindBuffer(GL_ARRAY_BUFFER, viz_cylinder_vbo);
	glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STATIC_DRAW);
	viz_cylinder_vertex_count = static_cast<GLsizei>(v.size() / 6);

	// One marker instance per field line (not per segment), at the start point
	// of its first segment (on the emitter surface), relative to the render origin
	vector<float> starts;

	for (const field_line& fl : lines)
		if (fl.is_first_segment)
			viz_push(starts, fl.start - origin);

	glBindBuffer(GL_ARRAY_BUFFER, viz_marker_instance_vbo);
	glBufferData(GL_ARRAY_BUFFER, starts.size() * sizeof(float), starts.data(), GL_STATIC_DRAW);
	viz_marker_instance_count = static_cast<GLsizei>(starts.size() / 3);

	viz_cylinder_built_count = lines.size();
	viz_cylinder_built_origin = origin;
	viz_cylinder_built_radius = static_cast<float>(radius);
}

void viz_draw_dynamic(const vector<float>& v, const GLenum mode, const float r, const float g, const float b, const float a)
{
	if (v.empty())
		return;

	glBindVertexArray(viz_dynamic_vao);
	glBindBuffer(GL_ARRAY_BUFFER, viz_dynamic_vbo);
	glBufferData(GL_ARRAY_BUFFER, v.size() * sizeof(float), v.data(), GL_STREAM_DRAW);
	glUniform4f(viz_color_location, r, g, b, a);
	glDrawArrays(mode, 0, static_cast<GLsizei>(v.size() / 3));
}

void viz_update_title(const render_snapshot& snap)
{
	char buffer[256];

	snprintf(buffer, sizeof(buffer),
		"Receiver (cyan) / forward (orange) boxes  |  step %zu of %zu  |  r = %.6g  |  field lines %zu%s",
		snap.step, snap.step_count,
		static_cast<real_type>(snap.receiver_box.r),
		recorded_field_line_count.load(std::memory_order_relaxed),
		snap.finished ? "  |  simulation finished" : "");

	if (viz_last_title != buffer)
	{
		viz_last_title = buffer;
		glutSetWindowTitle(buffer);
	}
}

void viz_display()
{
	render_snapshot snap;
	{
		std::lock_guard<std::mutex> lock(render_mutex);
		snap = shared_snapshot;
	}

	glClearColor(0.05f, 0.06f, 0.08f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (!snap.valid)
	{
		glutSwapBuffers();
		return;
	}

	const real_type box_size = snap.receiver_box.half_extents.length();

	if (!viz_camera_initialized)
	{
		viz_default_distance = static_cast<float>(6.0 * box_size);
		viz_camera_distance = viz_default_distance;
		viz_camera_initialized = true;
	}

	if (viz_sphere_vertex_count == 0)
		viz_build_emitter_sphere(snap.emitter_radius);

	viz_update_title(snap);

	// Everything is drawn relative to the midpoint of the two boxes
	vector_3 target = snap.receiver_box.center + snap.forward_box.center;
	target = target * static_cast<real_type>(0.5);

	// Orbit camera, +z up (the forward box is offset along world +z)
	const float cp = cosf(viz_camera_pitch);
	const vec3f eye =
	{
		viz_camera_distance * cp * cosf(viz_camera_yaw),
		viz_camera_distance * cp * sinf(viz_camera_yaw),
		viz_camera_distance * sinf(viz_camera_pitch)
	};

	const mat4f view = mat4f_look_at(eye, { 0, 0, 0 }, { 0, 0, 1 });

	const float aspect = static_cast<float>(viz_window_width) / static_cast<float>(viz_window_height > 0 ? viz_window_height : 1);
	const float z_near = viz_camera_distance * 0.01f;
	const float z_far = viz_camera_distance + static_cast<float>(4.0 * (snap.emitter_radius + snap.receiver_box.r));

	const mat4f view_proj = mat4f_multiply(mat4f_perspective(0.8f, aspect, z_near, z_far), view);

	glUseProgram(viz_program);
	glUniformMatrix4fv(viz_mvp_location, 1, GL_FALSE, view_proj.m);

	// --- Emitter wireframe (stored in world coordinates, so translate by -target) ---
	if (viz_show_emitter && viz_sphere_vertex_count > 0)
	{
		const vec3f offset =
		{
			static_cast<float>(-target.x),
			static_cast<float>(-target.y),
			static_cast<float>(-target.z)
		};

		const mat4f mvp = mat4f_multiply(view_proj, mat4f_translate(offset));
		glUniformMatrix4fv(viz_mvp_location, 1, GL_FALSE, mvp.m);
		glUniform4f(viz_color_location, 0.30f, 0.33f, 0.40f, 1.0f);
		glBindVertexArray(viz_sphere_vao);
		glDrawArrays(GL_LINES, 0, viz_sphere_vertex_count);

		glUniformMatrix4fv(viz_mvp_location, 1, GL_FALSE, view_proj.m);
	}

	// --- World axes at the receiver centre ---
	if (viz_show_axes)
	{
		vector_3 c = snap.receiver_box.center - target;
		const real_type len = 1.5 * box_size;

		vector<float> ax;
		viz_push(ax, c); viz_push(ax, c + vector_3(len, 0, 0));
		viz_draw_dynamic(ax, GL_LINES, 1.0f, 0.25f, 0.25f, 1.0f);

		ax.clear();
		viz_push(ax, c); viz_push(ax, c + vector_3(0, len, 0));
		viz_draw_dynamic(ax, GL_LINES, 0.25f, 1.0f, 0.25f, 1.0f);

		ax.clear();
		viz_push(ax, c); viz_push(ax, c + vector_3(0, 0, len));
		viz_draw_dynamic(ax, GL_LINES, 0.35f, 0.5f, 1.0f, 1.0f);
	}

	// --- Box edges (opaque) ---
	vector<float> receiver_edges, forward_edges;
	viz_build_OBB_edges(snap.receiver_box, target, receiver_edges);
	viz_build_OBB_edges(snap.forward_box, target, forward_edges);

	viz_draw_dynamic(receiver_edges, GL_LINES, 0.20f, 0.85f, 1.00f, 1.0f);
	viz_draw_dynamic(forward_edges, GL_LINES, 1.00f, 0.60f, 0.15f, 1.0f);

	const real_type field_line_radius = viz_field_line_radius_scale * box_size;
	const vec3f light_dir = vec3f_normalize(eye);

	if (viz_show_field_lines || viz_show_markers)
		viz_update_field_line_buffers(target, field_line_radius);

	// --- Field lines as thin cylinders (opaque, lit by a headlight) ---
	if (viz_show_field_lines)
	{
		if (viz_cylinder_vertex_count > 0)
		{
			glUseProgram(viz_lit_program);
			glUniformMatrix4fv(viz_lit_mvp_location, 1, GL_FALSE, view_proj.m);
			glUniform4f(viz_lit_color_location, 0.95f, 0.90f, 0.35f, 1.0f);
			glUniform3f(viz_lit_light_dir_location, light_dir.x, light_dir.y, light_dir.z);

			glBindVertexArray(viz_cylinder_vao);
			glDrawArrays(GL_TRIANGLES, 0, viz_cylinder_vertex_count);

			glUseProgram(viz_program);
		}
	}

	// --- Field line start markers: small spheres on the emitter surface ---
	if (viz_show_markers && viz_marker_instance_count > 0 && viz_marker_mesh_vertex_count > 0)
	{
		glUseProgram(viz_marker_program);
		glUniformMatrix4fv(viz_marker_mvp_location, 1, GL_FALSE, view_proj.m);
		glUniform1f(viz_marker_radius_location, static_cast<float>(viz_marker_radius_factor * field_line_radius));
		glUniform4f(viz_marker_color_location, 1.00f, 0.35f, 0.45f, 1.0f);
		glUniform3f(viz_marker_light_dir_location, light_dir.x, light_dir.y, light_dir.z);

		glBindVertexArray(viz_marker_vao);
		glDrawArraysInstanced(GL_TRIANGLES, 0, viz_marker_mesh_vertex_count, viz_marker_instance_count);

		glUseProgram(viz_program);
	}

	// --- Box faces (translucent, drawn last without depth writes) ---
	if (viz_show_faces)
	{
		vector<float> receiver_faces, forward_faces;
		viz_build_OBB_faces(snap.receiver_box, target, receiver_faces);
		viz_build_OBB_faces(snap.forward_box, target, forward_faces);

		glEnable(GL_BLEND);
		glDepthMask(GL_FALSE);

		viz_draw_dynamic(receiver_faces, GL_TRIANGLES, 0.20f, 0.85f, 1.00f, 0.15f);
		viz_draw_dynamic(forward_faces, GL_TRIANGLES, 1.00f, 0.60f, 0.15f, 0.15f);

		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
	}

	glBindVertexArray(0);
	glutSwapBuffers();
}

void viz_reshape(int width, int height)
{
	viz_window_width = width;
	viz_window_height = height;
	glViewport(0, 0, width, height);
}

void viz_timer(int)
{
	glutPostRedisplay();
	glutTimerFunc(33, viz_timer, 0); // ~30 fps
}

void viz_keyboard(unsigned char key, int, int)
{
	switch (key)
	{
	case 'f': case 'F': viz_show_faces = !viz_show_faces; break;
	case 'e': case 'E': viz_show_emitter = !viz_show_emitter; break;
	case 'a': case 'A': viz_show_axes = !viz_show_axes; break;
	case 'l': case 'L': viz_show_field_lines = !viz_show_field_lines; break;
	case 's': case 'S': viz_show_markers = !viz_show_markers; break;
	case '-': viz_marker_radius_factor /= 1.5f; break;
	case '=': case '+': viz_marker_radius_factor *= 1.5f; break;
	case '[': viz_field_line_radius_scale /= 1.5f; break;
	case ']': viz_field_line_radius_scale *= 1.5f; break;
	case 'r': case 'R':
		viz_camera_yaw = viz_default_yaw;
		viz_camera_pitch = viz_default_pitch;
		viz_camera_distance = viz_default_distance;
		break;
	case 27: // Esc
		glutLeaveMainLoop();
		return;
	}

	glutPostRedisplay();
}

void viz_mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		viz_dragging = (state == GLUT_DOWN);
		viz_last_mouse_x = x;
		viz_last_mouse_y = y;
	}
}

void viz_motion(int x, int y)
{
	if (!viz_dragging)
		return;

	viz_camera_yaw -= 0.01f * static_cast<float>(x - viz_last_mouse_x);
	viz_camera_pitch += 0.01f * static_cast<float>(y - viz_last_mouse_y);

	const float limit = 1.55f; // just under pi/2, avoids flipping over the pole
	if (viz_camera_pitch > limit) viz_camera_pitch = limit;
	if (viz_camera_pitch < -limit) viz_camera_pitch = -limit;

	viz_last_mouse_x = x;
	viz_last_mouse_y = y;

	glutPostRedisplay();
}

void viz_mouse_wheel(int, int direction, int, int)
{
	viz_camera_distance *= (direction > 0) ? (1.0f / 1.15f) : 1.15f;
	glutPostRedisplay();
}



int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitContextVersion(4, 0);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(viz_window_width, viz_window_height);

	// Return from glutMainLoop when the window is closed, so we can stop the simulation cleanly
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	glutCreateWindow("Receiver / forward boxes");

	glewExperimental = GL_TRUE; // required for core profiles
	const GLenum glew_status = glewInit();

	if (glew_status != GLEW_OK)
	{
		cout << "GLEW error: " << glewGetErrorString(glew_status) << endl;
		return 1;
	}

	glGetError(); // glewInit can leave a harmless GL_INVALID_ENUM behind in core profiles

	if (!viz_init_gl())
		return 1;

	glutDisplayFunc(viz_display);
	glutReshapeFunc(viz_reshape);
	glutKeyboardFunc(viz_keyboard);
	glutMouseFunc(viz_mouse);
	glutMotionFunc(viz_motion);
	glutMouseWheelFunc(viz_mouse_wheel);
	glutTimerFunc(33, viz_timer, 0);

	// The simulation runs in the background; GLUT must own the main thread
	std::thread simulation_thread(run_simulation);

	glutMainLoop();

	abort_requested.store(true, std::memory_order_relaxed);
	simulation_thread.join();

	return 0;
}