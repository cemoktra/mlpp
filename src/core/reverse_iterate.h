#ifndef _INCLUDED_REVERSE_ITERATE_H_
#define _INCLUDED_REVERSE_ITERATE_H_

template <typename T>
class reverse_range
{
    T &x;
    
public:
    reverse_range(T &x) : x(x) {}
    
    auto begin() const -> decltype(this->x.rbegin())
    {
        return x.rbegin();
    }
    
    auto end() const -> decltype(this->x.rend())
    {
        return x.rend();
    }
};
 
template <typename T>
reverse_range<T> reverse_iterate(T &x)
{
    return reverse_range<T>(x);
}

#endif