#include "naive_bayes.h"
#include "distribution.h"
#include <core/one_hot.h>
#include <iostream>

naive_bayes::naive_bayes(std::shared_ptr<distribution> distribution)
    : m_distribution(distribution)
{
}

Eigen::MatrixXd naive_bayes::predict(const Eigen::MatrixXd& x)
{    
    return m_distribution->predict(x);
}

double naive_bayes::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
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

void naive_bayes::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    auto y_onehot = (y.cols() == m_number_of_classes) ? y : one_hot::transform(y, m_number_of_classes);
    m_distribution->calc_weights(x, y_onehot);
}

void naive_bayes::init_classes(size_t number_of_classes)
{
    m_number_of_classes = number_of_classes;
}

void naive_bayes::set_weights(const Eigen::MatrixXd& weights)
{
    m_distribution->set_weights(weights);
}

Eigen::MatrixXd naive_bayes::weights()
{
    return m_distribution->weights();
}