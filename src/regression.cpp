#include "regression.h"

regression::regression(double rate, double threshold)
    : m_rate(rate)
    , m_threshold(threshold)
{
}

void regression::set_weights(const Eigen::MatrixXd& weights)
{
    m_weights = weights;
}

void regression::train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations)
{
    m_weights = calc_weights(x, y);
}

Eigen::MatrixXd regression::weights()
{
    return m_weights;
}

double regression::score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
{
    auto p = predict(x);
    auto e = y - p;
    auto mean = y.col(0).mean();
    auto ymean = y - Eigen::MatrixXd::Constant(y.rows(), y.cols(), mean);
    return 1.0 - (e.squaredNorm() / ymean.squaredNorm());
}