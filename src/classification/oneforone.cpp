#include "oneforone.h"
#include "logreg.h"
#include <iostream>

one_for_one::one_for_one() 
    : classifier()
{
    register_param("learning_rate", 0.02);
    register_param("threshold", 0.0001);
    register_param("max_iterations", 0);
}

one_for_one::~one_for_one()
{
    for (auto lr : m_models)
        delete lr;
}

Eigen::MatrixXd one_for_one::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd prediction_count = Eigen::MatrixXd::Constant(x.rows(), m_number_of_classes, 0);

    size_t model = 0;
    for (auto i = 0; i < m_number_of_classes - 1; i++) {
        for (auto j = i + 1; j < m_number_of_classes; j++) {
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
        for (auto i = 0; i < m_number_of_classes; i++) {
            if (prediction_count(k, i) > prediction_result(k, 0))
            {
                prediction_result(k, 0) = prediction_count(k, i);
                prediction_result(k, 1) = i;
            }
        }
    }
    return prediction_result;
}

void one_for_one::init_classes(size_t number_of_classes)
{
    m_number_of_classes = number_of_classes;
}

void one_for_one::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    for (auto lr : m_models)
        delete lr;

    for (auto i = 0; i < m_number_of_classes - 1; i++) {
        for (auto j = i + 1; j < m_number_of_classes; j++) {
            Eigen::MatrixXd y_, x_;
            for (auto k = 0; k < y.rows(); k++)
            {
                bool include_data = y.cols() > 1 ? 
                    y(k, i) > 0.0 || y(k, j) > 0.0 :
                    y(k, 0) == i || y(k, 0) == j;
                if (include_data)
                {                
                    if (x_.size()) {
                        x_.conservativeResize(x_.rows() + 1, Eigen::NoChange);
                        y_.conservativeResize(y_.rows() + 1, Eigen::NoChange);
                        x_.row(x_.rows() - 1) = x.row(k);
                        y_.row(y_.rows() - 1) = y.row(k);
                    } else {
                        x_ = x.row(k);
                        y_ = y.row(k);
                    }
                }
            }
            y_ = (y_.array() > 0.0).cast<double>();
            logistic_regression *lr = new logistic_regression();
            lr->set_param("learning_rate", get_param("learning_rate"));
            lr->set_param("threshold", get_param("threshold"));
            lr->set_param("max_iterations", get_param("max_iterations"));
            m_models.push_back(lr);
            lr->init_classes(m_number_of_classes);
            lr->train(x_, y_);
        }
    }
}

double one_for_one::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    size_t pos = 0, neg = 0;

    auto p = predict(x);
    for (auto i = 0; i < p.rows(); i++) {
        size_t predict_class = p(i, 1);
        size_t target_class = 0;
        auto yrow = y.row(i);
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

void one_for_one::set_weights(const Eigen::MatrixXd& weights)
{
    for (auto i = 0; i < weights.cols(); i++) {
        m_models[i]->set_weights(weights.col(i));
    }
}

Eigen::MatrixXd one_for_one::weights()
{
    Eigen::MatrixXd weights;

    for (auto i = 0; i < m_models.size(); i++) {
        if (i == 0)
            weights = m_models[i]->weights();
        else {
            weights.conservativeResize(Eigen::NoChange, weights.cols() + 1);
            weights.col(weights.cols() - 1) = m_models[i]->weights();
        }
    }

    return weights;
}