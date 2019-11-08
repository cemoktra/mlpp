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
            m_data[i] = _mm256_add_pd(m_data[i], rhs.m_data[i]);
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

void matrix::set_col(size_t col, const std::vector<double>& data)
{
    if (data.size() != m_cols)
        throw invalid_matrix_op();

    // TODO: create col iterator
    for (auto i = 0; i < data.size(); i++)
        set_at(i / m_cols, i % m_cols, data[i]);
}

std::pair<size_t, size_t> matrix::index_to_internal(size_t index)
{
    return std::make_pair(index / 4, index % 4);
}

double matrix::get_at(size_t row, size_t col)
{
    auto[block, offset] = index_to_internal(row * m_cols + col);        
    _mm256_store_pd(m_buffer, m_data[block]);
    return m_buffer[offset];
}

void matrix::set_at(size_t row, size_t col, double value)
{
    auto[block, offset] = index_to_internal(row * m_cols + col);        
    _mm256_store_pd(m_buffer, m_data[block]);
    m_buffer[offset] = value;
    m_data[block] = _mm256_load_pd(m_buffer);
}