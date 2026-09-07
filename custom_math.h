#ifndef custom_math_h
#define custom_math_h



typedef long double real_type;



#include <algorithm>
using std::sort;

#include <limits>
using std::numeric_limits;

#include <vector>
using std::vector;

#include <set>
using std::set;

#include <map>
using std::map;


#include <iostream>
using std::cout;
using std::endl;

#include <cmath>


namespace custom_math
{
	class vector_3;
	class line_segment_3;
	class circle_3;

	class indexed_triangle;
	class sorted_indexed_triangle;
	class sorted_indexed_edge;
	class indexed_ngon;

	class indexed_curved_triangle;

	class vector_4;
	class circle_4;
	class line_segment_4;

	const real_type pi = 3.14159265358979323846;
	const real_type pi_half = pi / 2;
	const real_type pi_2 = 2 * pi;
	const real_type epsilon = 1e-6;

	real_type d(const real_type& a, const real_type& b);
	real_type d_3(const vector_3& a, const vector_3& b);
	real_type d_3_sq(const vector_3& a, const vector_3& b);
	real_type d_4(const vector_4& a, const vector_4& b);
};

class custom_math::vector_3
{
public:
	real_type x, y, z;

	inline bool operator<(const vector_3& right) const
	{
		if (right.x > x)
			return true;
		else if (right.x < x)
			return false;

		if (right.y > y)
			return true;
		else if (right.y < y)
			return false;

		if (right.z > z)
			return true;
		else if (right.z < z)
			return false;

		return false;
	}

	vector_3(const real_type& src_x = 0, const real_type& src_y = 0, const real_type& src_z = 0);
	bool operator==(const vector_3& rhs);
	bool operator!=(const vector_3& rhs);
	void zero(void);
	void rotate_x(const real_type& radians);
	void rotate_y(const real_type& radians);
	void rotate_z(const real_type& radians);
	vector_3 operator+(const vector_3& rhs);
	vector_3 operator-(const vector_3& rhs) const;
	vector_3 operator*(const vector_3& rhs);
	vector_3 operator/(const vector_3& rhs);

	vector_3 operator*(const real_type& rhs);
	vector_3 operator/(const real_type& rhs);
	vector_3& operator=(const vector_3& rhs);
	vector_3& operator+=(const vector_3& rhs);
	vector_3& operator*=(const vector_3& rhs);
	vector_3& operator*=(const real_type& rhs);
	vector_3 operator-(void);
	real_type length(void) const;
	vector_3& normalize(void);
	real_type dot(const vector_3& rhs) const;
	real_type self_dot(void) const;
	vector_3 cross(const vector_3& rhs) const;
};


class custom_math::vector_4
{
public:
	real_type x, y, z, w;

	vector_4(const real_type& src_x = 0, const real_type& src_y = 0, const real_type& src_z = 0, const real_type& src_w = 0);
	void zero(void);
	vector_4 operator+(const vector_4& rhs);
	vector_4 operator-(const vector_4& rhs);
	vector_4 operator*(const vector_4& rhs);
	vector_4 operator*(const real_type& rhs);
	vector_4 operator/(const real_type& rhs);
	vector_4& operator=(const vector_4& rhs);
	vector_4& operator+=(const vector_4& rhs);
	vector_4& operator*=(const vector_4& rhs);
	vector_4& operator*=(const real_type& rhs);
	vector_4 operator-(void);
	real_type length(void) const;
	vector_4& normalize(void);
	real_type dot(const vector_4& rhs) const;
	real_type self_dot(void) const;
};


class custom_math::line_segment_3
{
public:
	vector_3 start, end;

	real_type length(void)
	{
		return d_3(start, end);
	}

	bool operator<(line_segment_3& rhs)
	{
		return length() < rhs.length();
	}
};

class custom_math::line_segment_4
{
public:
	vector_4 start, end;

	real_type length(void)
	{
		return d_4(start, end);
	}

	bool operator<(line_segment_4& rhs)
	{
		return length() < rhs.length();
	}
};


class custom_math::circle_3
{
public:
	vector_3 U, V;

	void get_vertices(size_t num_steps, real_type radius, vector<vector_3>& vertices)
	{
		vertices.clear();

		for (size_t step = 0; step < num_steps; step++)
		{
			const real_type circumference_arc = 2 * pi / static_cast<real_type>(num_steps);
			real_type t = step * circumference_arc;
			static vector_3 v;
			v = U * cos(t) + V * sin(t);

			real_type vlen = v.length();

			if (vlen > 1.0 + epsilon || vlen < 1.0 - epsilon)
				cout << "circle_3 parameterization error: " << vlen << endl;

			v = v * radius;

			vertices.push_back(v);
		}
	}

	void make_Vy_zero(void)
	{
		if (U.y > 0.0 - epsilon && U.y < 0.0 + epsilon)
		{
			static vector_3 temp_v;
			temp_v = U;
			U = V;
			V = temp_v;
		}
	}

	// Note: These reparameterization functions work with vectors of arbitrary dimension.
	// e.g., No 3D cross product operations were used.
	void reparameterize_U(void)
	{
		// TODO: Handle special case where U is close to 0 or 1 (rectangle goes to zero area).
		// TODO: Make north pole the U vector?

		// Make a rectangle.
		static vector_3 U1, U2, U3, U4;

		U.normalize();

		U1 = U;

		// Invert all, and then revert y back to original sign.
		U2 = -U;
		U2.y = -U2.y;

		// Invert all.
		U3 = -U;

		// Invert y.
		U4 = U;
		U4.y = -U4.y;

		U = ((U1 - U3) + (U4 - U2)).normalize();
		V = ((U1 - U3) + (U2 - U4)).normalize();

		make_Vy_zero();

		if (U.y < 0)
			U.y = -U.y;
	}

	void reparameterize_UV(void)
	{
		// Make a rectangle.
		static vector_3 U1, U2, U3, U4;

		U.normalize();
		V.normalize();

		// Handle special case where the vertices are antipodal.
		real_type dot = U.dot(V);

		if (dot < -1.0 + epsilon)
		{
			cout << "circle_3 -- handling antipodal reparameterization." << endl;
			V = vector_3(0, 1, 0);
		}

		U1 = U * cos(0.0) + V * sin(0.0);
		U2 = U * cos(0.5 * pi) + V * sin(0.5 * pi);
		U3 = U * cos(pi) + V * sin(pi);
		U4 = U * cos(1.5 * pi) + V * sin(1.5 * pi);

		U = ((U1 - U3) + (U4 - U2)).normalize();
		V = ((U1 - U3) + (U2 - U4)).normalize();

		make_Vy_zero();
	}
};


class custom_math::circle_4
{
public:
	vector_4 U, V;

	void get_vertices(size_t num_steps, real_type radius, vector<vector_4>& vertices)
	{
		vertices.clear();

		for (size_t step = 0; step < num_steps; step++)
		{
			const real_type circumference_arc = 2 * pi / static_cast<real_type>(num_steps);
			real_type t = step * circumference_arc;
			static vector_4 v;
			v = U * cos(t) + V * sin(t);

			real_type vlen = v.length();

			if (vlen > 1.0 + epsilon || vlen < 1.0 - epsilon)
				cout << "circle_4 parameterization error: " << vlen << endl;

			v = v * radius;

			vertices.push_back(v);
		}
	}

	void make_Vy_zero(void)
	{
		if (U.y > 0.0 - epsilon && U.y < 0.0 + epsilon)
		{
			static vector_4 temp_v;
			temp_v = U;
			U = V;
			V = temp_v;
		}
	}

	// Note: These reparameterization functions work with vectors of arbitrary dimension.
	// e.g., No 3D cross product operations were used.
	void reparameterize_U(void)
	{
		// TODO: Handle special case where U is close to 0 or 1 (rectangle goes to zero area).
		// TODO: Make north pole the U vector?

		// Make a rectangle.
		static vector_4 U1, U2, U3, U4;

		U.normalize();

		U1 = U;

		// Invert all, and then revert y back to original sign.
		U2 = -U;
		U2.y = -U2.y;

		// Invert all.
		U3 = -U;

		// Invert y.
		U4 = U;
		U4.y = -U4.y;

		U = ((U1 - U3) + (U4 - U2)).normalize();
		V = ((U1 - U3) + (U2 - U4)).normalize();

		make_Vy_zero();

		if (U.y < 0)
			U.y = -U.y;
	}

	void reparameterize_UV(void)
	{
		// Make a rectangle.
		static vector_4 U1, U2, U3, U4;

		U.normalize();
		V.normalize();

		// Handle special case where the vertices are antipodal.
		real_type dot = U.dot(V);

		if (dot < -1.0 + epsilon)
		{
			cout << "circle_4 -- handling antipodal reparameterization." << endl;
			V = vector_4(0, 1, 0, 0);
		}

		U1 = U * cos(0.0) + V * sin(0.0);
		U2 = U * cos(0.5 * pi) + V * sin(0.5 * pi);
		U3 = U * cos(pi) + V * sin(pi);
		U4 = U * cos(1.5 * pi) + V * sin(1.5 * pi);

		U = ((U1 - U3) + (U4 - U2)).normalize();
		V = ((U1 - U3) + (U2 - U4)).normalize();

		make_Vy_zero();
	}
};



class custom_math::sorted_indexed_edge
{
public:
	sorted_indexed_edge(const size_t src_v0, const size_t src_v1)
	{
		if (src_v0 < src_v1)
		{
			v0 = src_v0;
			v1 = src_v1;
		}
		else
		{
			v1 = src_v0;
			v0 = src_v1;
		}
	}

	bool operator==(const sorted_indexed_edge& rhs) const
	{
		// vertices should be pre-sorted
		if (v0 == rhs.v0 && v1 == rhs.v1)
			return true;

		return false;
	}

	bool operator<(const sorted_indexed_edge& rhs) const
	{
		// vertices should be pre-sorted
		if (v0 < rhs.v0)
			return true;
		else if (v0 > rhs.v0)
			return false;
		// else first components are equal

		if (v1 < rhs.v1)
			return true;

		return false;
	}

	size_t v0, v1;
};

class custom_math::indexed_triangle
{
public:
	size_t i0, i1, i2; // vertices
};








#endif

