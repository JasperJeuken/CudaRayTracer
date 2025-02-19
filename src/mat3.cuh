/**
 * @file mat3.cuh
 * @author Jasper Jeuken
 * @brief Defines a 3x3 matrix class
 */
#ifndef MAT3_H
#define MAT3_H

#include <iostream>
#include "vec3.cuh"

/**
 * @class mat3
 * @brief 3x3 matrix
 */
class mat3 {
public:
    float m[3][3]; ///< Matrix data

    /**
     * @brief Default constructor
     * 
     * @return Zero matrix
     */
    __host__ __device__ mat3() {
        m[0][0] = 0; m[0][1] = 0; m[0][2] = 0;
        m[1][0] = 0; m[1][1] = 0; m[1][2] = 0;
        m[2][0] = 0; m[2][1] = 0; m[2][2] = 0;
    }

    /**
     * @brief Construct a new mat3 object from 9 values
     * 
     * @param[in] m00 Value at row 0, column 0
     * @param[in] m01 Value at row 0, column 1
     * @param[in] m02 Value at row 0, column 2
     * @param[in] m10 Value at row 1, column 0
     * @param[in] m11 Value at row 1, column 1
     * @param[in] m12 Value at row 1, column 2
     * @param[in] m20 Value at row 2, column 0
     * @param[in] m21 Value at row 2, column 1
     * @param[in] m22 Value at row 2, column 2
     * @return Constructed matrix
     */
    __host__ __device__ mat3(float m00, float m01, float m02, 
                             float m10, float m11, float m12, 
                             float m20, float m21, float m22) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }

    /**
     * @brief Construct a new mat3 object from 3 column vectors
     * 
     * @param[in] col1 First column vector
     * @param[in] col2 Second column vector
     * @param[in] col3 Third column vector
     * @return Constructed matrix
     */
    __host__ __device__ mat3(vec3 col1, vec3 col2, vec3 col3) {
        m[0][0] = col1[0]; m[0][1] = col2[0]; m[0][2] = col3[0];
        m[1][0] = col1[1]; m[1][1] = col2[1]; m[1][2] = col3[1];
        m[2][0] = col1[2]; m[2][1] = col2[2]; m[1][2] = col3[2];
    }

    /**
     * @brief Construct a new identity matrix
     * 
     * @return Identity matrix
     */
    __host__ __device__ static mat3 identity() {
        return mat3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    }

    /**
     * @brief Multiply the matrix with a vector
     * 
     * @param[in] u Vector to multiply with
     * @return Resulting vector
     */
    __host__ __device__ vec3 operator*(const vec3& u) const {
        return vec3(m[0][0] * u.x() + m[0][1] * u.y() + m[0][2] * u.z(),
                    m[1][0] * u.x() + m[1][1] * u.y() + m[1][2] * u.z(),
                    m[2][0] * u.x() + m[2][1] * u.y() + m[2][2] * u.z());
    }

    /**
     * @brief Multiply the matrix with another matrix
     * 
     * @param[in] other Matrix to multiply with
     * @return Resulting matrix
     */
    __host__ __device__ mat3 operator*(const mat3& other) const {
        mat3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result.m[i][j] = m[i][0] * other.m[0][j] +
                                 m[i][1] * other.m[1][j] +
                                 m[i][2] * other.m[2][j];
            }
        }
        return result;
    }

    /**
     * @brief Create a rotation matrix from euler angles
     * 
     * @param[in] angles Euler angles (radians)
     * @return Constructed rotation matrix 
     */
    __host__ __device__ static mat3 from_euler(const vec3& angles) {
        float cx = cos(angles.x());
        float sx = sin(angles.x());
        float cy = cos(angles.y());
        float sy = sin(angles.y());
        float cz = cos(angles.z());
        float sz = sin(angles.z());

        mat3 rot_x = mat3(1, 0, 0, 0, cx, -sx, 0, sx, cx);
        mat3 rot_y = mat3(cy, 0, sy, 0, 1, 0, -sy, 0, cy);
        mat3 rot_z = mat3(cz, -sz, 0, sz, cz, 0, 0, 0, 1);

        return rot_z * rot_y * rot_x;
    }

    /**
     * @brief Transpose the matrix
     * 
     * @return Transposed matrix 
     */
    __host__ __device__ mat3 transpose() const {
        return mat3(m[0][0], m[1][0], m[2][0],
                    m[0][1], m[1][1], m[2][1],
                    m[0][2], m[1][2], m[2][2]);
    }

    /**
     * @brief Invert the matrix
     * 
     * @return Inverted matrix 
     */
    __host__ __device__ mat3 inverse() const {
        float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        float inv_det = 1.0f / det;

        mat3 result;
        result.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        result.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        result.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        result.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        result.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        result.m[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det;
        result.m[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det;
        result.m[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det;
        result.m[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det;

        return result;
    }
};

/**
 * @brief Print 3x3 matrix
 * 
 * @param[in] out Output stream
 * @param[in] m Matrix to print
 * @return Output stream
 */
inline std::ostream& operator<<(std::ostream& out, const mat3& m) {
    out << "[ " << m.m[0][0] << " " << m.m[0][1] << " " << m.m[0][2] << "  \n"
        << "  " << m.m[1][0] << " " << m.m[1][1] << " " << m.m[1][2] << "  \n"
        << "  " << m.m[2][0] << " " << m.m[2][1] << " " << m.m[2][2] << " ]";
    return out;
}

#endif