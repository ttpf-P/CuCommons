#ifndef EM_CUDA_VEC3

#ifdef EM_VEC3_COMMONS
#error "Include CUDA commons before C++ commons"
#endif

#define EM_CUDA_VEC3
#define EM_VEC3_COMMONS

#define __common__ __host__ __device__

#include <cmath>
#include "CUrng.hpp"

using std::sqrt;

class vec3 {
public:
    __common__ vec3() : e{0,0,0} {}
    __common__ vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    [[nodiscard]] __common__ double x() const { return e[0]; }
    [[nodiscard]] __common__ double y() const { return e[1]; }
    [[nodiscard]] __common__ double z() const { return e[2]; }

    __common__ vec3 operator-() const {return {-e[0], -e[1], -e[2]};}
    __common__ double operator[](int i) const {return e[i];}
    __common__ double& operator[](int i) {return e[i];}

    __common__ vec3& operator+=(const vec3 &v){
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __common__ vec3& operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __common__ vec3& operator/=(const double t) {
        return *this *=(1/t);
    }

    [[nodiscard]] __common__ double length() const {
        return sqrt(length_squared());
    }

    [[nodiscard]] __common__ double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __device__ static vec3 random(unsigned long long &state) {
        return vec3(random_double(state), random_double(state), random_double(state));
    }

    __device__ static vec3 random(double min, double max, unsigned long long &state) {
        return vec3(random_double(min,max,state), random_double(min,max,state), random_double(min,max,state));
    }

    inline static vec3 random() {
        return vec3(random_double(), random_double(), random_double());
    }

    inline static vec3 random(double min, double max) {
        return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
    }

    __common__ bool near_zero() const {
        const auto epsilon = 1e-8;
        return (fabs(e[0]) < epsilon) && (fabs(e[1]) < epsilon) && (fabs(e[2]) < epsilon);
    }

public:
    double e[3];

};


// aliases
using point3 = vec3;
using color = vec3;

// funcs
__common__ inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__common__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return {u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}

__common__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return {u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
}

__common__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return {u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
}

__common__ inline vec3 operator*(double t, const vec3 &v) {
    return {t*v.e[0], t*v.e[1], t*v.e[2]};
}

__common__ inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

__common__ inline vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

__common__ inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
           + u.e[1] * v.e[1]
           + u.e[2] * v.e[2];
}

__common__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
            u.e[2] * v.e[0] - u.e[0] * v.e[2],
            u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

__common__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__device__ vec3 random_in_unit_sphere(unsigned long long &state) {
    while (true) {
        auto p = vec3::random(-1,1,state);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}
__device__ vec3 random_unit_vector(unsigned long long &state) {
    return unit_vector(random_in_unit_sphere(state));
}

vec3 random_in_unit_sphere() {
    while (true) {
        auto p = vec3::random(-1,1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}
vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

__common__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__common__ vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ vec3 random_in_unit_disk(unsigned long long &state) {
    while (true) {
        auto p = vec3(random_double(-1,1,state), random_double(-1,1,state), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(random_double(-1,1), random_double(-1,1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}


#endif
