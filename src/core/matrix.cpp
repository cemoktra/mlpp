#include "matrix.h"
#include <math.h>
#include <cstring>

// _mm256_load_pd
// _mm256_store_pd
// _mm256_mul_pd
// ...

matrix::matrix(size_t rows, size_t cols)
    : m_rows(rows)
    , m_cols(cols)
{
    m_int_size = ceil((rows * cols) / 4.0);
    m_data = new __m256d[m_int_size];
}

matrix::matrix(const matrix& rhs)
    : m_rows(rhs.m_rows)
    , m_cols(rhs.m_cols)
    , m_int_size(rhs.m_int_size)
{
    m_data = new __m256d[m_int_size];
    memcpy(m_data, rhs.m_data, m_int_size * sizeof(__m256d));
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
        // TODO: create iterator
        for (auto i = 0; i < m_int_size; i++)
        {
            _mm256_add_pd(m_data[i], rhs.m_data[i]);
        }
        return *this;
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