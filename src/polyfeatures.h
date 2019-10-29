#ifndef _POLYFEATURES_H_
#define _POLYFEATURES_H_

#include <Eigen/Dense>

class polynomial_features
{
public:
    polynomial_features(size_t degree = 2, bool bias = false);
    ~polynomial_features() = default;

    Eigen::MatrixXd transform(const Eigen::MatrixXd &x);

private:
    size_t m_degree;
    bool m_bias;
};

#endif