#include "multinomial_logreg.h"

#include <iostream>

multinomial_logistic_regression::multinomial_logistic_regression() 
    : logistic_regression()
{
}

Eigen::MatrixXd multinomial_logistic_regression::predict(const Eigen::MatrixXd& x) 
{
    Eigen::MatrixXd z = x * m_weights;
    return softmax(z);
}

Eigen::MatrixXd multinomial_logistic_regression::softmax(const Eigen::MatrixXd& z)
{
    Eigen::MatrixXd result (z.rows(), z.cols());
    Eigen::MatrixXd e = z.array().exp();

    for (auto i = 0; i < e.rows(); i++)
    {
        auto row = e.row(i);
        result.row(i) = row / row.sum();
    }
    return result;
}

double multinomial_logistic_regression::cost(const Eigen::MatrixXd& y, const Eigen::MatrixXd& p)
{
    Eigen::MatrixXd logp = p.array().log();
    return -y.cwiseProduct(logp).sum() / y.rows();
}