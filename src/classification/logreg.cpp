#include "logreg.h"
#include <core/one_hot.h>
#include <stdexcept>
#include <iostream>

logistic_regression::logistic_regression() 
    : classifier()
{
    register_param("learning_rate", 0.02);
    register_param("threshold", 0.0001);
    register_param("max_iterations", 0);
}

Eigen::MatrixXd logistic_regression::predict(const Eigen::MatrixXd& x) 
{
    if (x.cols() == m_weights.rows()) {
        return sigmoid(x * m_weights);
    }
    throw std::invalid_argument("x dimension is wrong");
}

void logistic_regression::init_classes(const std::vector<std::string>& classes)
{
    m_classes = classes;
}

void logistic_regression::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    auto y_onehot = (y.cols() == m_classes.size()) ? y : one_hot::transform(y, m_classes);
    calc_weights(x, y_onehot);
}

void logistic_regression::set_weights(const Eigen::MatrixXd& weights)
{
    m_weights = weights;
}

Eigen::MatrixXd logistic_regression::weights()
{
    return m_weights;
}

double logistic_regression::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
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


void logistic_regression::calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    double current_cost, last_cost;
    m_weights = Eigen::MatrixXd::Ones(x.cols(), y.cols());
    size_t max_iterations = static_cast<size_t>(get_param("max_iterations"));
    size_t iteration = 0;
    last_cost = std::numeric_limits<double>::max();

    while (true) {
        auto p = predict(x);
        auto g = gradient(x, y, p);
        m_weights -= g;

        current_cost = cost(y, p);
        if (last_cost - current_cost < get_param("threshold"))
            break;
        last_cost = current_cost;
        iteration++;
        if (max_iterations > 0 && iteration >= max_iterations)
            break;
    }
}

double logistic_regression::cost(const Eigen::MatrixXd& y, const Eigen::MatrixXd& p)
{
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(y.rows(), y.cols());
    Eigen::MatrixXd p1, logp, logp1, c1, c2;

    p1 = ones - p;
    logp  = p.array().log();
    logp1 = p1.array().log();
    c1 = (-y).cwiseProduct(logp);
    c2 = (ones - y).cwiseProduct(logp1);
    return (c1 - c2).sum() / p.rows();
}

Eigen::MatrixXd logistic_regression::gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& p)
{
    Eigen::MatrixXd gradient = x.transpose() * (p - y);
    return get_param("learning_rate") * gradient / x.rows();
}


Eigen::MatrixXd logistic_regression::sigmoid(const Eigen::MatrixXd& x)
{    
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(x.rows(), x.cols());
    Eigen::MatrixXd e = (-x).array().exp();
    Eigen::MatrixXd q = e.array() + 1;
    Eigen::MatrixXd result = 1.0 / q.array();
    if ((result.array() < 0.0).any() || (result.array() > 1.0).any())
        throw std::out_of_range("ERROR IN SIGMOID");    
    return result;
}