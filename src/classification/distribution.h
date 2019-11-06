#ifndef _DISTRIBUTION_H_
#define _DISTRIBUTION_H_

#include <Eigen/Dense>

class distribution
{
public:
    distribution() = default;
    distribution(const distribution&) = delete;
    ~distribution() = default;

    virtual void calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) = 0;
    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) = 0;
    
    virtual Eigen::MatrixXd weights() = 0;
    virtual void set_weights(const Eigen::MatrixXd& weights) = 0;
};

#endif