#ifndef _TRAINTEST_H_
#define _TRAINTEST_H_

#include <Eigen/Dense>

class test_train
{
public:
    test_train() = delete;
    ~test_train() = delete;

    static void split(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test, Eigen::MatrixXd& y_train, Eigen::MatrixXd& y_test, double test_proportion = 0.25, bool shuffle  = true);
};

#endif