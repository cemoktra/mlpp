#ifndef _LINREG_H_
#define _LINREG_H_

#include "regression.h"

class linear_regression : public regression
{
public:
    linear_regression();
    ~linear_regression() = default;

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;

protected:
    Eigen::MatrixXd calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) override;
};

#endif