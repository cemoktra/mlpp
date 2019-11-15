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

    matrix_iterator(pointer ptr, size_t step = 1) : m_ptr(ptr), m_step(step) { }
    matrix_iterator(const matrix_iterator& rhs) : m_ptr(rhs.m_ptr), m_step(rhs.m_step) { }

    inline self_type operator++() { self_type i = *this; m_ptr += m_step; return i; };
    inline self_type operator++(int) { m_ptr += m_step; return *this; };
    inline reference operator*() { return *m_ptr; }
    inline pointer operator->() { return m_ptr; }
    inline bool operator==(const self_type& rhs) { return m_ptr == rhs.m_ptr; };
    inline bool operator!=(const self_type& rhs) { return m_ptr != rhs.m_ptr; };

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

    matrix_avx_iterator(double *srcPtr, double *endPtr = nullptr, size_t step = 1) 
        : m_srcPtr(srcPtr) 
        , m_endPtr(endPtr)
        , m_step(step)
    {
        if (!endPtr)
            m_endPtr = m_srcPtr;

        update();
    }
    matrix_avx_iterator(const matrix_avx_iterator& rhs)
        : m_srcPtr(rhs.m_srcPtr)
        , m_endPtr(rhs.m_endPtr)
        , m_value(rhs.m_value)
        , m_step(rhs.m_step)
        , m_int_values(rhs.m_int_values)
    {}

    inline self_type operator++() {
        save();
        self_type i = std::move(*this);
        m_srcPtr += m_int_values;
        update();
        return i;
    }
    self_type operator++(int) {
        save();
        m_srcPtr += m_int_values;
        update();
        return std::move(*this);
    }
    inline reference operator*() { return m_value; }
    inline pointer operator->() { return &m_value; }
    inline bool operator==(const self_type& rhs) { return m_srcPtr == rhs.m_srcPtr; }
    inline bool operator!=(const self_type& rhs) { return !(*this == rhs); }

    inline void save() { _mm256_storeu_pd(m_srcPtr, m_value); }

private:
    inline void update() { m_value = _mm256_loadu_pd(m_srcPtr); m_int_values = std::min(4l, m_endPtr - m_srcPtr); };

    double __attribute__ ((aligned (4))) m_buffer[4];

    double *m_srcPtr;
    double *m_endPtr;
    value_type m_value;
    size_t m_step;
    size_t m_int_values;
};

#endif