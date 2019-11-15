#include "matrix.h"
#include <math.h>
#include <cstring>
#include <algorithm>
#include <execution>

matrix::matrix(size_t rows, size_t cols)
    : m_rows(rows)
    , m_cols(cols)
{
    m_data = new double[rows * cols];
}

matrix::matrix(const matrix& rhs)
    : m_rows(rhs.m_rows)
    , m_cols(rhs.m_cols)
{
    m_data = new double[m_rows * m_cols];
    memcpy(m_data, rhs.m_data, m_rows * m_cols * sizeof(double));
}

matrix::~matrix()
{
    delete [] m_data;
}

const matrix matrix::operator+(const matrix& rhs)
{
    matrix copy(*this);
    copy += rhs;
    return copy;
}

const matrix& matrix::operator+=(const matrix& rhs)
{
    if (rhs.m_cols == m_cols && rhs.m_rows == m_rows) {
        //std::transform(begin(), end(), rhs.begin(), begin(), [&](const double& a, const double& b) { return a + b; });
        auto e = end();
        for (auto it1 = begin(), it2 = rhs.begin(); it1 != e; it1++, it2++)
        {
            *it1 = *it1 + *it2;
        }
        return *this;
    }
    else if (rhs.m_cols == 1 && rhs.m_rows == m_rows) {
        for (auto c = 0; c < cols(); c++) {
            // std::transform(col_begin(c), col_end(c), rhs.begin(), col_begin(c), [&](const double& a, const double& b) { return a + b; });
            auto e = col_end(c);
            for (auto it1 = col_begin(c), it2 = rhs.begin(); it1 != e; it1++, it2++)
            {
                *it1 = *it1 + *it2;
            }
        }
        return *this;
    }
    else if (m_cols == 1 && rhs.m_rows == m_rows) {
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
    }
    else if (m_rows == 1 && rhs.m_cols == m_cols) {
    }

    throw invalid_matrix_op();
}

matrix_iterator matrix::begin() const
{
    return matrix_iterator(m_data);
}

matrix_iterator matrix::end() const
{
    return matrix_iterator(m_data + (m_rows * m_cols));
}

matrix_iterator matrix::row_begin(size_t row) const
{
    return matrix_iterator(&m_data[row * m_cols]);
}

matrix_iterator matrix::row_end(size_t row) const
{
    return matrix_iterator(&m_data[row * m_cols + m_cols]);
}

matrix_iterator matrix::col_begin(size_t col) const
{
    return matrix_iterator(&m_data[col], m_cols);
}

matrix_iterator matrix::col_end(size_t col) const
{
    return matrix_iterator(&m_data[m_rows * m_cols + col], m_cols);
}

matrix_avx_iterator matrix::avx_begin() const
{
    return matrix_avx_iterator(m_data, m_data + (m_rows * m_cols));
}

matrix_avx_iterator matrix::avx_end() const
{
    return matrix_avx_iterator(m_data + (m_rows * m_cols));
}

double matrix::get_at(size_t row, size_t col)
{
    return m_data[row * m_cols + col];
}

void matrix::set_at(size_t row, size_t col, double value)
{
    m_data[row * m_cols + col] = value;
}

void matrix::avx_add(const matrix& rhs)
{
    if (rhs.m_cols == m_cols && rhs.m_rows == m_rows) {
        // std::transform(std::execution::par, avx_begin(), avx_end(), rhs.avx_begin(), avx_begin(), [&](const __m256d& a, const __m256d& b) { return _mm256_add_pd(a, b); });
        auto end = avx_end();
        for (auto it1 = avx_begin(), it2 = rhs.avx_begin(); it1 != end; it1++, it2++)
        {
            *it1 = _mm256_add_pd(*it1, *it2);
        }
        return;
    }
    else if (rhs.m_cols == 1 && rhs.m_rows == m_rows) {
    }
    else if (m_cols == 1 && rhs.m_rows == m_rows) {
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
    }
    else if (m_rows == 1 && rhs.m_cols == m_cols) {
    }

    throw invalid_matrix_op();
}

