#ifndef _MATRIX_ITERATOR_H_
#define _MATRIX_ITERATOR_H_

#include <iterator>

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

#endif