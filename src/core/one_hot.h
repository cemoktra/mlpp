#ifndef _ONEHOT_H_
#define _ONEHOT_H_

#include <Eigen/Dense>

class one_hot
{
public:
    one_hot() = delete;
    ~one_hot() = delete;

    static Eigen::MatrixXd transform(const Eigen::MatrixXd& x);
};

#endif