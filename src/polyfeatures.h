#ifndef _POLYFEATURES_H_
#define _POLYFEATURES_H_

#include <vector>

class polynomial_features
{
public:
    polynomial_features(size_t degree = 2, bool bias = false);
    ~polynomial_features() = default;

    std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &x);

private:
    size_t m_degree;
    bool m_bias;
};

#endif