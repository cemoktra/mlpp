#include "gauss_distribution.h"
#include <core/pseudo_inverse.h>

void gauss_distribution::calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    m_mean.resize(x.cols(), y.cols());
    m_var.resize(x.cols(), y.cols());
    m_pre_prop.resize(y.cols());

    for (auto c = 0; c < y.cols(); c++) {
        m_mean.col(c) = (x.array().colwise() * y.col(c).array()).colwise().sum() / y.col(c).sum();
        m_var.col(c)  = ((x.array().rowwise() - m_mean.col(c).transpose().array()).square().colwise() * y.col(c).array()).colwise().sum() / y.col(c).sum();
        m_pre_prop(c) = y.col(c).sum() / y.rows();
    }
}

Eigen::MatrixXd gauss_distribution::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd pfc (x.rows(), m_pre_prop.size());
    Eigen::MatrixXd pcf (x.rows(), m_pre_prop.size());
    Eigen::VectorXd total_prop = Eigen::VectorXd::Zero(x.rows());

    for (auto c = 0; c < m_pre_prop.size(); c++) {
        pfc.col(c) = calc_pfc(x, c);
        total_prop = total_prop.array() + (pfc.col(c).array() * m_pre_prop(c));
    }

    for (auto c = 0; c < m_pre_prop.size(); c++)
        pcf.col(c) = (pfc.col(c).array() * m_pre_prop(c)) / total_prop.array();
    return pcf;
}

Eigen::MatrixXd gauss_distribution::weights()
{
    Eigen::MatrixXd weights (2 * m_mean.rows() + 1, m_mean.cols());
    weights.row(0) = m_pre_prop;
    weights.block(1, 0, m_mean.rows(), m_mean.cols()) = m_mean;
    weights.block(1 + m_mean.rows(), 0, m_var.rows(), m_var.cols()) = m_var;
    return weights;
}

void gauss_distribution::set_weights(const Eigen::MatrixXd& weights)
{
    m_pre_prop = weights.row(0);
    m_mean = weights.block(1, 0, (weights.rows() - 1) / 2, weights.cols());
    m_var = weights.block(1 + m_mean.rows(), 0, (weights.rows() - 1) / 2, weights.cols());
}

Eigen::VectorXd gauss_distribution::calc_pfc(const Eigen::MatrixXd& x, size_t _class)
{
    static const double pi = 3.14159265359;
    Eigen::VectorXd product = Eigen::VectorXd::Ones(x.rows());
    Eigen::MatrixXd denom = (m_var.col(_class).array() * 2.0 * pi).sqrt();    
    Eigen::MatrixXd sqr = (x.array().rowwise() - m_mean.col(_class).transpose().array()).square();
    Eigen::MatrixXd var2_inv = pseudo_inverse(2.0 *  m_var.col(_class).transpose());
    Eigen::MatrixXd e = (sqr * var2_inv * -0.5).array().exp();

    for (auto feature = 0; feature < x.cols(); feature++)
        product = product.cwiseProduct(e / denom(feature, 0));
    return product;
}