#include "binomial_distribution.h"
#include <core/pseudo_inverse.h>
#include <iostream>

void binomial_distribution::calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    m_pre_prop.resize(y.cols());
    m_smooth.resize(y.cols());
    m_feature_prop.resize(x.cols(), y.cols());

    for (auto c = 0; c < y.cols(); c++) {
        Eigen::VectorXd class_features = (x.array().colwise() * y.col(c).array()).colwise().sum();
        class_features = class_features.array() + 1;
        class_features = class_features.array() / (y.col(c).sum() + y.rows());
        class_features = class_features.array().log();
        m_feature_prop.col(c) = class_features;

        m_pre_prop(c) = y.col(c).sum() / y.rows();
        m_smooth(c) = log(1.0 / (y.col(c).sum() + y.rows()));
    }    
}

Eigen::MatrixXd binomial_distribution::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd result (x.rows(), m_pre_prop.size());

    for (auto r = 0; r < x.rows(); r++) {
        result.row(r) = m_pre_prop;

        for (auto c = 0; c < m_pre_prop.size(); c++) {
            for (auto feature = 0; feature < x.cols(); feature++) {
                if (x(r, feature) > 0.0) {
                    if (m_feature_prop(feature, c) > 0.0)
                        result(r, c) += (x(r, feature) * m_feature_prop(feature, c));
                    else {
                        bool add_smoothing = false;
                        for (auto c2 = 0; c2 < m_pre_prop.size(); c2++) {
                            if (c2 == c)
                                continue;
                            if (m_feature_prop(feature, c2) > 0.0)
                                add_smoothing = true;
                        }
                        if (add_smoothing)
                            result(r, c) += (x(r, feature) * m_smooth(c));
                    }
                }
            }
        }
    }
    return result;
}

Eigen::MatrixXd binomial_distribution::weights()
{
    Eigen::MatrixXd weights (m_feature_prop.rows() + 2, m_feature_prop.cols());
    weights.row(0) = m_pre_prop;
    weights.row(1) = m_smooth;
    weights.block(2, 0, m_feature_prop.rows(), m_feature_prop.cols()) = m_feature_prop;
    return weights;
}

void binomial_distribution::set_weights(const Eigen::MatrixXd& weights)
{
    m_pre_prop = weights.row(0);
    m_smooth = weights.row(1);
    m_feature_prop = weights.block(2, 0, weights.rows() - 2, weights.cols());
}
