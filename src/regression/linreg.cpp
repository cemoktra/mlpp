#include "linreg.h"
#include <stdexcept>

linear_regression::linear_regression() 
    : m_rate(0.0002)
    , m_threshold(0.0001)
{
}

Eigen::MatrixXd linear_regression::predict(const Eigen::MatrixXd& x) 
{
    if (x.cols() == m_weights.rows()) {
        return x * m_weights;
    }
    throw std::invalid_argument("x dimension is wrong");
}

void linear_regression::train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations)
{
    calc_weights(x, y);
}


double linear_regression::score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
{
    auto p = predict(x);
    auto e = y - p;
    auto mean = y.col(0).mean();
    auto ymean = y - Eigen::MatrixXd::Constant(y.rows(), y.cols(), mean);
    return 1.0 - (e.squaredNorm() / ymean.squaredNorm());
}

void linear_regression::set_weights(const Eigen::MatrixXd& weights)
{
    m_weights = weights;
}

Eigen::MatrixXd linear_regression::weights()
{
    return m_weights;
}

void linear_regression::calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
{
    Eigen::MatrixXd xTx = (x.transpose() * x);
    Eigen::MatrixXd xTy = (x.transpose() * y);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(xTx, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd xTx_inverse = svd.solve(Eigen::MatrixXd::Identity(xTx.rows(), xTx.cols()));
    m_weights = xTx_inverse * xTy;
}
