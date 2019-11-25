#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "matrix_iterator.h"
#include <exception>
#include <immintrin.h>

class invalid_matrix_op : std::exception
{
public:
    invalid_matrix_op() : std::exception() {}

    const char* what() const noexcept override { return "invalid matrix operation! (wrong dimensions)"; }
};


class matrix 
{
public:
    matrix(size_t rows, size_t cols);
    matrix(const matrix& rhs);
    matrix(matrix&& rhs);
    ~matrix();

    matrix& operator=(const matrix& rhs);
    matrix& operator=(matrix&& rhs);
    matrix& operator=(const double& rhs);

    const matrix  operator+(const matrix& rhs);
    const matrix& operator+=(const matrix& rhs);
    const matrix  operator-(const matrix& rhs);
    const matrix& operator-=(const matrix& rhs);
    const matrix  operator*(const matrix& rhs);
    const matrix& operator*=(const matrix& rhs);
    const matrix  operator/(const matrix& rhs);
    const matrix& operator/=(const matrix& rhs);

    inline virtual size_t cols() const { return m_cols; };
    inline virtual size_t rows() const { return m_rows; };

    virtual matrix_iterator begin() const;
    virtual matrix_iterator end() const;

    virtual matrix_iterator row_begin(size_t row) const;
    virtual matrix_iterator row_end(size_t row) const;

    virtual matrix_iterator col_begin(size_t col) const;
    virtual matrix_iterator col_end(size_t col) const;

    virtual double get_at(size_t row, size_t col) const;
    virtual void set_at(size_t row, size_t col, double value);

    matrix matmul(const matrix& rhs);
    matrix transpose();

    void exp();

private:
    alignas(16) double *m_data;

    size_t m_rows;
    size_t m_cols;
};

#endif