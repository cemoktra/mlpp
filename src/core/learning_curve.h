#ifndef _LEARNING_CURVE_H_
#define _LEARNING_CURVE_H_

#include <Eigen/Dense>

class model;

class learning_curve
{
public:
    learning_curve() = default;
    learning_curve(const learning_curve&) = delete;
    ~learning_curve() = default;

    static Eigen::MatrixXd create(model *m, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);
};

#endif