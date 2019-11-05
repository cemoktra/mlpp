#include "naive_bayes_gauss.h"
#include <core/one_hot.h>
#include <iostream>

naive_bayes_gauss::naive_bayes_gauss()
{
}

Eigen::MatrixXd naive_bayes_gauss::predict(const Eigen::MatrixXd& x)
{    
    Eigen::MatrixXd m_pfc (x.rows(), m_classes.size());
    Eigen::MatrixXd m_pcf (x.rows(), m_classes.size());
    Eigen::VectorXd v_total_prop = Eigen::VectorXd::Zero(x.rows());

    
    for (auto c = 0; c < m_classes.size(); c++) {
        m_pfc.col(c) = pfc(x, c);
        v_total_prop = v_total_prop.array() + (m_pfc.col(c).array() * m_pre_prop(c));
    }

    for (auto c = 0; c < m_classes.size(); c++)
        m_pcf.col(c) = (m_pfc.col(c).array() * m_pre_prop(c)) / v_total_prop.array();
    return m_pcf;
}

double naive_bayes_gauss::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    size_t pos = 0, neg = 0;
    auto p = predict(x);

    for (auto i = 0; i < p.rows(); i++) {
        auto prow = p.row(i);
        auto yrow = y.row(i);
        std::vector<double> pvec (prow.data(), prow.data() + prow.rows() * prow.cols());
        size_t predict_class = std::max_element(pvec.begin(), pvec.end()) - pvec.begin();
        size_t target_class = 0;
        if (y.cols() > 1) {
            std::vector<double> yvec (yrow.data(), yrow.data() + yrow.rows() * yrow.cols());
            target_class = std::max_element(yvec.begin(), yvec.end()) - yvec.begin();
        } else
            target_class = static_cast<size_t>(yrow(0));
        
        if (predict_class == target_class)
            pos++;
        else
            neg++;        
    }
    return static_cast<double>(pos) / static_cast<double>(pos + neg);
}

void naive_bayes_gauss::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    auto y_onehot = (y.cols() == m_classes.size()) ? y : one_hot::transform(y, m_classes);

    m_mean.resize(x.cols(), y_onehot.cols());
    m_var.resize(x.cols(), y_onehot.cols());
    m_pre_prop.resize(y_onehot.cols());

    for (auto c = 0; c < y_onehot.cols(); c++) {
        m_mean.col(c) = mean(x, y_onehot, c);
        m_var.col(c) = variance(x, y_onehot, c);
        m_pre_prop(c) = y_onehot.col(c).sum() / y.rows();
    }    
}

Eigen::VectorXd naive_bayes_gauss::mean(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, size_t _class)
{
    return (x.array().colwise() * y.col(_class).array()).colwise().sum() / y.col(_class).sum();
}

Eigen::VectorXd naive_bayes_gauss::variance(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, size_t _class)
{
    return ((x.array().rowwise() - m_mean.col(_class).transpose().array()).square().colwise() * y.col(_class).array()).colwise().sum() / y.col(_class).sum();
}

Eigen::MatrixXd naive_bayes_gauss::pinv(const Eigen::MatrixXd& x)
{
    auto svd = x.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &singularValues = svd.singularValues();
    Eigen::MatrixXd singularValuesInv(x.cols(), x.rows());
    singularValuesInv.setZero();
    for (unsigned int i = 0; i < singularValues.size(); ++i)
        singularValuesInv(i, i) = singularValues(i) > 1e-6 ? 1.0 / singularValues(i) : 0.0;
    return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
}

Eigen::VectorXd naive_bayes_gauss::pfc(const Eigen::MatrixXd& x, size_t _class)
{
    static const double pi = 3.14159265359;
    Eigen::VectorXd product = Eigen::VectorXd::Ones(x.rows());
    Eigen::MatrixXd denom = (m_var.col(_class).array() * 2.0 * pi).sqrt();    
    Eigen::MatrixXd sqr = (x.array().rowwise() - m_mean.col(_class).transpose().array()).square();
    Eigen::MatrixXd var2_inv = pinv(2.0 *  m_var.col(_class).transpose());
    Eigen::MatrixXd e = (sqr * var2_inv * -0.5).array().exp();

    for (auto feature = 0; feature < x.cols(); feature++)
        product = product.cwiseProduct(e / denom(feature, 0));
    return product;
}


void naive_bayes_gauss::init_classes(const std::vector<std::string>& classes)
{
    m_classes = classes;
}

void naive_bayes_gauss::set_weights(const Eigen::MatrixXd& weights)
{
    m_pre_prop = weights.row(0);
    m_mean = weights.block(1, 0, (weights.rows() - 1) / 2, weights.cols());
    m_var = weights.block(1 + m_mean.rows(), 0, (weights.rows() - 1) / 2, weights.cols());
}

Eigen::MatrixXd naive_bayes_gauss::weights()
{
    Eigen::MatrixXd weights (2 * m_mean.rows() + 1, m_mean.cols());
    weights.row(0) = m_pre_prop;
    weights.block(1, 0, m_mean.rows(), m_mean.cols()) = m_mean;
    weights.block(1 + m_mean.rows(), 0, m_var.rows(), m_var.cols()) = m_var;
    return weights;
}