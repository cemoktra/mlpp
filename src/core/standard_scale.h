#ifndef _STANDARD_SCALE_H_
#define _STANDARD_SCALE_H_

#include <Eigen/Dense>

class standard_scale
{
public:
    standard_scale() = delete;
    ~standard_scale() = delete;

    static Eigen::MatrixXd transform(const Eigen::MatrixXd& x);
};

#endif