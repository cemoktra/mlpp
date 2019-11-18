#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "matrix_iterator.h"
#include <exception>
#include <vector>


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
    ~matrix();

    const matrix  operator+(const matrix& rhs);
    const matrix& operator+=(const matrix& rhs);
    const matrix  operator*(const matrix& rhs);
    const matrix& operator*=(const matrix& rhs);

    inline size_t cols() const { return m_cols; };

    matrix_iterator begin() const;
    matrix_iterator end() const;

    matrix_iterator row_begin(size_t row) const;
    matrix_iterator row_end(size_t row) const;

    matrix_iterator col_begin(size_t col) const;
    matrix_iterator col_end(size_t col) const;

    matrix_avx_iterator avx_begin() const;
    matrix_avx_iterator avx_end() const;

    double get_at(size_t row, size_t col);
    void set_at(size_t row, size_t col, double value);

    void avx_add(const matrix& rhs);
    void avx_mul(const matrix& rhs);

private:
    typedef double aligned_double __attribute__ ((aligned (32)));

    aligned_double *m_data;
    // double *m_data;

    size_t m_rows;
    size_t m_cols;
};


#endif