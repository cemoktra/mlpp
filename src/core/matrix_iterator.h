#ifndef _MATRIX_ITERATOR_H_
#define _MATRIX_ITERATOR_H_

#include <iterator>
#include <immintrin.h>

class matrix_iterator
{
public:
    typedef matrix_iterator self_type;
    typedef double value_type;
    typedef double& reference;
    typedef double* pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef int difference_type;

    matrix_iterator(pointer ptr, size_t step = 1);
    matrix_iterator(const matrix_iterator& rhs);

    self_type operator++();
    self_type operator++(int);
    reference operator*();
    pointer operator->();
    bool operator==(const self_type& rhs);
    bool operator!=(const self_type& rhs);

private:
    pointer m_ptr;
    size_t m_step;
};

class matrix_avx_iterator
{
public:
    typedef matrix_avx_iterator self_type;
    typedef __m256d value_type;
    typedef __m256d& reference;
    typedef __m256d* pointer;
    typedef std::forward_iterator_tag iterator_category;
    typedef int difference_type;

    matrix_avx_iterator(double *srcPtr, double *endPtr = nullptr, size_t step = 1);
    matrix_avx_iterator(const matrix_avx_iterator& rhs);

    self_type operator++();
    self_type operator++(int);
    reference operator*();
    pointer operator->();
    bool operator==(const self_type& rhs);
    bool operator!=(const self_type& rhs);

    void save();

private:
    void update();

    double __attribute__ ((aligned (4))) m_buffer[4];

    double *m_srcPtr;
    double *m_endPtr;
    value_type m_value;
    size_t m_step;
    size_t m_int_values;
};

#endif