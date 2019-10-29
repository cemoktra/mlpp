#include "logreg.h"
#include <stdexcept>
#include <iostream>

logistic_regression::logistic_regression() 
    : regression(0.02, 0.0001)
{
}

Eigen::MatrixXd logistic_regression::predict(const Eigen::MatrixXd& x) 
{
    if (x.cols() == m_weights.rows()) {
        return sigmoid(x * m_weights);
    }
    throw std::invalid_argument("x dimension is wrong");
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


Eigen::MatrixXd logistic_regression::calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
{
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(y.rows(), y.cols());
    Eigen::MatrixXd p, p1, logp, logp1, c1, c2, gradient;
    double cost;
    auto samples = x.rows();
    m_weights = Eigen::MatrixXd::Ones(x.cols(), 1);

    while (true) {
        p  = predict(x);
        p1 = ones - p;
        logp  = p.array().log();
        logp1 = p1.array().log();

        c1 = (-y).cwiseProduct(logp);
        c2 = (ones - y).cwiseProduct(logp1);
        cost = (c1 - c2).sum() / samples;

        gradient = x.transpose() * (p - y);
        gradient = m_rate * gradient / samples;
        m_weights -= gradient;
        
        if (!(gradient.array().abs() > m_threshold).any())
            break;
    }

    return m_weights;
}

double logistic_regression::score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
{
    size_t pos = 0, neg = 0;

    auto p = predict(x);
    
    for (auto i = 0; i < p.rows(); i++) {
        double bin_result = p(i, 0) > 0.5 ? 1.0 : 0.0;
        if (bin_result == y(i, 0))
            pos++;
        else
            neg++;        
    }
    return static_cast<double>(pos) / static_cast<double>(pos + neg);
}