#include "oneforall.h"
#include "logreg.h"
#include <iostream>

one_for_all::one_for_all(const std::map<std::string, double>& class_map)
    : model_interface()
    , m_class_map(class_map)
{
}

one_for_all::~one_for_all()
{
    for (auto lr : m_models)
        delete lr;
}

Eigen::MatrixXd one_for_all::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd prediction_result (x.rows(), 2);
    prediction_result.col(0) = Eigen::MatrixXd::Zero(x.rows(), 1);
    prediction_result.col(1) = Eigen::MatrixXd::Constant(x.rows(), 1, -1);

    for (auto j = 0; j < m_models.size(); j++) {
        auto p = m_models[j]->predict(x);
        for (auto i = 0; i < x.rows(); i++) {
            if (p(i, 0) > prediction_result(i, 0)) {
                prediction_result(i, 0) = p(i, 0);
                prediction_result(i, 1) = m_class_values[j];
            }
        }
    }
    return prediction_result;
}

void one_for_all::train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations)
{
    for (auto lr : m_models)
        delete lr;

    m_class_values = std::vector<double> (y.data(), y.data() + y.rows() * y.cols());
    std::sort(m_class_values.begin(), m_class_values.end());
    m_classes = std::unique(m_class_values.begin(), m_class_values.end()) - m_class_values.begin();

    for (auto i = 0; i < m_classes; i++) {
        Eigen::MatrixXd y_ = (y.array() == m_class_values[i]).cast<double>();
        
        logistic_regression *lr = new logistic_regression();
        m_models.push_back(lr);
        lr->train(x, y_);
    }
}

double one_for_all::score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
{
    size_t pos = 0, neg = 0;

    auto p = predict(x);
    
    for (auto i = 0; i < p.rows(); i++) {
        if (p(i, 1) == y(i, 0))
            pos++;
        else
            neg++;        
    }
    return static_cast<double>(pos) / static_cast<double>(pos + neg);
}

std::string one_for_all::mapped_class(double value)
{
    for (auto c : m_class_map) {
        if (c.second == value)
            return c.first;
    }
    return "";
}