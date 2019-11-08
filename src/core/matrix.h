#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <immintrin.h>
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

    void set_col(size_t col, const std::vector<double>& data);

    double get_at(size_t row, size_t col);
    void set_at(size_t row, size_t col, double value);

private:
    std::pair<size_t, size_t> index_to_internal(size_t index);

    double __attribute__ ((aligned (4))) m_buffer[4];
    __m256d *m_data;

    size_t m_int_size;
    size_t m_rows;
    size_t m_cols;
};


#endif