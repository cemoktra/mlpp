#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <immintrin.h>
#include <exception>

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



private:
    __m256d *m_data;

    size_t m_int_size;
    size_t m_rows;
    size_t m_cols;
};


#endif