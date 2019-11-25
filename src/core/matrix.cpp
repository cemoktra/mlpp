#include "matrix.h"
#include <math.h>
#include <cstring>
#include <algorithm>
#include <execution>
#include <xsimd/xsimd.hpp>
#include <cblas.h>

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
{
    m_rows = std::exchange(rhs.m_rows, 0);
    m_cols = std::exchange(rhs.m_cols, 0);
    m_data = std::exchange(rhs.m_data, nullptr);
}

matrix::~matrix()
{
    delete [] m_data;
}

matrix& matrix::operator=(const matrix& rhs)
{
    if (&rhs != this) {
        m_rows = rhs.m_rows;
        m_cols = rhs.m_cols;
        m_data = new double[m_rows * m_cols];
        memcpy(m_data, rhs.m_data, m_rows * m_cols * sizeof(double));
    }
    return *this;
}

matrix& matrix::operator=(matrix&& rhs)
{
    if (&rhs != this) {
        delete [] m_data;
        m_rows = std::exchange(rhs.m_rows, 0);
        m_cols = std::exchange(rhs.m_cols, 0);
        m_data = std::exchange(rhs.m_data, nullptr);
    }
    return *this;
}

matrix& matrix::operator=(const double& rhs)
{
    std::fill(begin(), end(), rhs);
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
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a + *rhs_row; });
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        for (auto row = 0; row < rows(); ++row)
            std::transform(std::execution::par, row_begin(row), row_end(row), rhs.begin(), row_begin(row), std::plus<>());
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
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a - *rhs_row; });
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        for (auto row = 0; row < rows(); ++row)
            std::transform(std::execution::par, row_begin(row), row_end(row), rhs.begin(), row_begin(row), std::minus<>());
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
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a * *rhs_row; });
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        for (auto row = 0; row < rows(); ++row)
            std::transform(std::execution::par, row_begin(row), row_end(row), rhs.begin(), row_begin(row), std::multiplies<>());
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
            std::transform(std::execution::par, row_begin(row), row_end(row), row_begin(row), [&](const double& a) { return a / *rhs_row; });
    }
    else if (rhs.m_rows == 1 && rhs.m_cols == m_cols) {
        for (auto row = 0; row < rows(); ++row)
            std::transform(std::execution::par, row_begin(row), row_end(row), rhs.begin(), row_begin(row), std::divides<>());
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

matrix matrix::matmul(const matrix& rhs)
{
    if (rhs.m_rows != m_cols)
        throw invalid_matrix_op();
    
    matrix result (rows(), rhs.cols());
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows(), rhs.cols(), cols(), 1.0, m_data, cols(), rhs.m_data, rhs.cols(), 1.0, result.m_data, rhs.cols());
    return std::move(result);
}

matrix matrix::transpose()
{
    matrix result (cols(), rows());
    for (auto i = 0; i < rows(); i++)
        std::copy(row_begin(i), row_end(i), col_begin(i));
    return std::move(result);
}

void matrix::exp()
{
    using b_type = xsimd::simd_type<double>;
    std::size_t inc = b_type::size;
    std::size_t size = rows() * cols();
    std::size_t vec_size = size - size % inc;
    for(std::size_t i = 0; i < vec_size; i +=inc)
    {        
        b_type avec = xsimd::load_aligned(&m_data[i]);
        avec = xsimd::exp(avec);
        avec.store_unaligned(&m_data[i]);
    }
    for(std::size_t i = vec_size; i < size; ++i)
        m_data[i] = std::exp(m_data[i]);
}