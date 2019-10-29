#include "linreg.h"
#include <stdexcept>

linear_regression::linear_regression() 
    : regression(0.0002, 0.0001)
{
}

Eigen::MatrixXd linear_regression::predict(const Eigen::MatrixXd& x) 
{
    if (x.cols() == m_weights.rows()) {
        return x * m_weights;
    }
    throw std::invalid_argument("x dimension is wrong");
}

Eigen::MatrixXd linear_regression::calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
{
    auto xTx = (x.transpose() * x);
    auto xTy = (x.transpose() * y);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(xTx, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto xTx_inverse = svd.solve(Eigen::MatrixXd::Identity(xTx.rows(), xTx.cols()));
    return xTx_inverse * xTy;
}