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

matrix::matrix(matrix&& rhs)
    : m_data(std::move(rhs.m_data))
    , m_rows(rhs.m_rows)
    , m_cols(rhs.m_cols)
{
}

matrix::~matrix()
{
    delete [] m_data;
}

matrix& matrix::operator=(const double& val)
{
    std::fill(begin(), end(), val);
    return *this;
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
        std::transform(std::execution::par, begin(), end(), rhs.begin(), begin(), std::plus<>());
    }
    else if (rhs.m_cols == 1 && rhs.m_rows == m_rows) {
        const auto rhs_end = rhs.end();
        size_t row = 0;
        for (auto rhs_row = rhs.begin(); rhs_row != rhs_end; ++rhs_row, ++row)
        {
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a + *rhs_row; });
        }
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        const auto rhs_end = rhs.end();
        size_t col = 0;
        for (auto rhs_col = rhs.begin(); rhs_col != rhs_end; ++rhs_col, ++col)
        {
            std::transform(std::execution::par, col_begin(col), col_end(col), col_begin(col), [&](const double& a) { return a + *rhs_col; });
        }
    }
    else
        throw invalid_matrix_op();
    
    return std::move(*this);
}

const matrix matrix::operator-(const matrix& rhs)
{
    matrix copy(*this);
    copy -= rhs;
    return copy;
}

const matrix& matrix::operator-=(const matrix& rhs)
{
    if (rhs.m_cols == m_cols && rhs.m_rows == m_rows) {
        std::transform(std::execution::par, begin(), end(), rhs.begin(), begin(), std::minus<>());
    }
    else if (rhs.m_cols == 1 && rhs.m_rows == m_rows) {
        const auto rhs_end = rhs.end();
        size_t row = 0;
        for (auto rhs_row = rhs.begin(); rhs_row != rhs_end; ++rhs_row, ++row)
        {
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a - *rhs_row; });
        }
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        for (auto r = 0; r < rows(); r++) {
            std::transform(std::execution::par, row_begin(r), row_end(r), rhs.begin(), row_begin(r), std::minus<>());
        }
    }
    else
        throw invalid_matrix_op();
    
    return std::move(*this);
}

const matrix matrix::operator*(const matrix& rhs)
{
    matrix copy(*this);
    copy *= rhs;
    return copy;
}

const matrix& matrix::operator*=(const matrix& rhs)
{
    if (rhs.m_cols == m_cols && rhs.m_rows == m_rows) {
        std::transform(std::execution::par, begin(), end(), rhs.begin(), begin(), std::multiplies<>());
    }
    else if (rhs.m_cols == 1 && rhs.m_rows == m_rows) {
        const auto rhs_end = rhs.end();
        size_t row = 0;
        for (auto rhs_row = rhs.begin(); rhs_row != rhs_end; ++rhs_row, ++row)
        {
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a * *rhs_row; });
        }
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        for (auto r = 0; r < rows(); r++) {
            std::transform(std::execution::par, row_begin(r), row_end(r), rhs.begin(), row_begin(r), std::multiplies<>());
        }
    }
    else
        throw invalid_matrix_op();
    return std::move(*this);
}

const matrix matrix::operator/(const matrix& rhs)
{
    matrix copy(*this);
    copy /= rhs;
    return copy;
}

const matrix& matrix::operator/=(const matrix& rhs)
{
    if (rhs.m_cols == m_cols && rhs.m_rows == m_rows) {
        std::transform(std::execution::par, begin(), end(), rhs.begin(), begin(), std::divides<>());
    }
    else if (rhs.m_cols == 1 && rhs.m_rows == m_rows) {
        const auto rhs_end = rhs.end();
        size_t row = 0;
        for (auto rhs_row = rhs.begin(); rhs_row != rhs_end; ++rhs_row, ++row)
        {
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a / *rhs_row; });
        }
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        for (auto r = 0; r < rows(); r++) {
            std::transform(std::execution::par, row_begin(r), row_end(r), rhs.begin(), row_begin(r), std::divides<>());
        }
    }
    else
        throw invalid_matrix_op();
    return std::move(*this);
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

double matrix::get_at(size_t row, size_t col) const
{
    return m_data[row * m_cols + col];
}

void matrix::set_at(size_t row, size_t col, double value)
{
    m_data[row * m_cols + col] = value;
}