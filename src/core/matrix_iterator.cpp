#include "matrix_iterator.h"
#include <iostream>

matrix_iterator::matrix_iterator(pointer ptr, size_t step) 
    : m_ptr(ptr) 
    , m_step(step)
{
}

matrix_iterator::matrix_iterator(const matrix_iterator& rhs)
{
    m_ptr = rhs.m_ptr;
    m_step = rhs.m_step;
}

matrix_iterator::self_type matrix_iterator::operator++() 
{ 
    self_type i = *this;
    m_ptr += m_step;
    return i;
}

matrix_iterator::self_type matrix_iterator::operator++(int)
{ 
    m_ptr += m_step; 
    return *this;
}

matrix_iterator::reference matrix_iterator::operator*()
{ 
    return *m_ptr;
}

matrix_iterator::pointer matrix_iterator::operator->() {
    return m_ptr;
}

bool matrix_iterator::operator==(const self_type& rhs)
{ 
    return m_ptr == rhs.m_ptr;
}

bool matrix_iterator::operator!=(const self_type& rhs)
{   
    return m_ptr != rhs.m_ptr;
}




matrix_avx_iterator::matrix_avx_iterator(double *srcPtr, double *endPtr, size_t step) 
    : m_srcPtr(srcPtr) 
    , m_endPtr(endPtr)
    , m_step(step)
{
    if (!endPtr)
        m_endPtr = m_srcPtr;

    update();
}

matrix_avx_iterator::matrix_avx_iterator(const matrix_avx_iterator& rhs)
{
    m_srcPtr = rhs.m_srcPtr;
    m_endPtr = rhs.m_endPtr;
    m_value = rhs.m_value;
    m_step = rhs.m_step;
    m_int_values = rhs.m_int_values;
}

void matrix_avx_iterator::update()
{
    m_int_values = 0;
    if (m_step == 1) {
        m_value = _mm256_loadu_pd(m_srcPtr);
        m_int_values = std::min(4l, m_endPtr - m_srcPtr);
    } else {
        while (m_int_values < 4 && m_srcPtr != m_endPtr) {
            m_buffer[m_int_values++] = *(m_srcPtr);
            m_srcPtr += m_step;
        }
        m_value = _mm256_load_pd(m_buffer); 
        m_srcPtr -= m_int_values * m_step;
    }
}

matrix_avx_iterator::self_type matrix_avx_iterator::operator++() 
{ 
    save();
    self_type i = *this;
    m_srcPtr += (m_int_values * m_step);
    update();
    return i;
}

matrix_avx_iterator::self_type matrix_avx_iterator::operator++(int)
{ 
    save();
    m_srcPtr += (m_int_values * m_step);
    update();
    return *this;
}

matrix_avx_iterator::reference matrix_avx_iterator::operator*()
{ 
    return m_value;
}

matrix_avx_iterator::pointer matrix_avx_iterator::operator->() {
    return &m_value;
}

bool matrix_avx_iterator::operator==(const self_type& rhs)
{ 
    return m_srcPtr == rhs.m_srcPtr;
}

bool matrix_avx_iterator::operator!=(const self_type& rhs)
{   
    return !(*this == rhs);
}

void matrix_avx_iterator::save()
{
    if (m_step == 1) {
        _mm256_storeu_pd(m_srcPtr, m_value);
    } else {
        _mm256_store_pd(m_buffer, m_value);
        for (auto i = 0; i < m_int_values; i++) 
            *(m_srcPtr + i * m_step) = m_buffer[i];
    }
}