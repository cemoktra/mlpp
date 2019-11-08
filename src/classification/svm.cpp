#include "svm.h"
#include <iostream>

svm::svm()
{
    register_param("c", 1.0);
    register_param("learning_rate", 0.02);
    register_param("threshold", 0.0001);
    register_param("max_iterations", 0);
}

Eigen::MatrixXd svm::predict(const Eigen::MatrixXd& x)
{
    return x * m_weights;
}

double svm::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
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

void svm::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    size_t max_iterations = static_cast<size_t>(get_param("max_iterations"));
    size_t iteration = 0;
    double current_cost, last_cost;
    m_weights = Eigen::MatrixXd::Ones(x.cols(), y.cols());
    last_cost = std::numeric_limits<double>::max();

    while (true) {
        auto p = predict(x);
        auto prod = p.array() * y.array();

        current_cost = 0;
        for (auto c = 0; c < prod.cols(); c++) 
        {    
            for (auto r = 0; r < prod.rows(); r++)
            {
                if (prod(r, c) >= 1.0) {
                    m_weights.col(c) = m_weights.col(c).array() - (get_param("learning_rate") * 2.0 * (1.0 / (iteration + 1)) * m_weights.col(c).array());
                }
                else {
                    current_cost += 1 - prod(r, c);
                    auto xy = (x.row(r).array() * y(r, c)).transpose();
                    m_weights.col(c) = m_weights.col(c).array() + (get_param("learning_rate") * 2.0 * xy.array() * (1.0 / (iteration + 1)) * m_weights.col(c).array());
                }     
            }
        }
        current_cost /= (y.cols() * y.rows());

        if (last_cost - current_cost < get_param("threshold"))
            break;
        last_cost = current_cost;
        iteration++;
        if (max_iterations > 0 && iteration >= max_iterations)
            break;
    }
}

void svm::init_classes(size_t number_of_classes)
{
}

void svm::set_weights(const Eigen::MatrixXd& weights)
{
}

Eigen::MatrixXd svm::weights()
{
    return Eigen::MatrixXd();
}