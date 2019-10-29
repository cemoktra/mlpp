#include "oneforone.h"
#include "logreg.h"
#include <iostream>

one_for_one::one_for_one(const std::map<std::string, double>& class_map)
    : model_interface()
    , m_class_map(class_map)
{
}

one_for_one::~one_for_one()
{
    for (auto lr : m_models)
        delete lr;
}

Eigen::MatrixXd one_for_one::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd prediction_count = Eigen::MatrixXd::Constant(x.rows(), m_classes, 0);

    size_t model = 0;
    for (auto i = 0; i < m_classes - 1; i++) {
        for (auto j = i + 1; j < m_classes; j++) {
            auto p = m_models[model]->predict(x);

            for (auto k = 0; k < x.rows(); k++) {
                if (p(k, 0) > 0.5)
                    prediction_count(k, i) += 1.0;
                else
                    prediction_count(k, j) += 1.0;
            }

            model++;
        }
    }

    Eigen::MatrixXd prediction_result (x.rows(), 2);
    prediction_result.col(0) = Eigen::MatrixXd::Zero(x.rows(), 1);
    prediction_result.col(1) = Eigen::MatrixXd::Constant(x.rows(), 1, -1);

    for (auto k = 0; k < x.rows(); k++) {
        for (auto i = 0; i < m_classes; i++) {
            if (prediction_count(k, i) > prediction_result(k, 0))
            {
                prediction_result(k, 0) = prediction_count(k, i);
                prediction_result(k, 1) = m_class_values[i];
            }
        }
    }
    return prediction_count;
}

void one_for_one::train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations)
{
    for (auto lr : m_models)
        delete lr;

    m_class_values = std::vector<double> (y.data(), y.data() + y.rows() * y.cols());
    std::sort(m_class_values.begin(), m_class_values.end());
    m_classes = std::unique(m_class_values.begin(), m_class_values.end()) - m_class_values.begin();

    for (auto i = 0; i < m_classes - 1; i++) {
        for (auto j = i + 1; j < m_classes; j++) {
            Eigen::MatrixXd y_, x_;
            for (auto k = 0; k < y.rows(); k++)
            {
                if (y(k, 0) == m_class_values[i] || y(k, 0) == m_class_values[j])
                {                
                    if (x_.size()) {
                        x_.conservativeResize(x_.rows() + 1, Eigen::NoChange);
                        x_.row(x_.rows() - 1) = x.row(k);
                        y_.conservativeResize(y_.rows() + 1, Eigen::NoChange);
                        y_.row(y_.rows() - 1) = y.row(k);
                    } else {
                        x_ = x.row(k);
                        y_ = y.row(k);
                    }
                }
            }
            y_ = (y_.array() == m_class_values[i]).cast<double>();
            logistic_regression *lr = new logistic_regression();
            m_models.push_back(lr);
            lr->train(x_, y_);
        }
    }
}

double one_for_one::score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y)
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

std::string one_for_one::mapped_class(double value)
{
    for (auto c : m_class_map) {
        if (c.second == value)
            return c.first;
    }
    return "";
}
