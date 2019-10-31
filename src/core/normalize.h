#ifndef _NORMALIZE_H_
#define _NORMALIZE_H_

#include <Eigen/Dense>

class normalize
{
public:
    normalize() = delete;
    ~normalize() = delete;

    static Eigen::MatrixXd transform(const Eigen::MatrixXd& x);
};

#endif