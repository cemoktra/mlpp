#ifndef _MULTINOMIALLOGREG_H_
#define _MULTINOMIALLOGREG_H_

#include "logreg.h"
#include <functional>

class multinomial_logistic_regression : public logistic_regression
{
public:
    multinomial_logistic_regression();
    ~multinomial_logistic_regression() = default;

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;

protected:
    Eigen::MatrixXd softmax(const Eigen::MatrixXd& z);

    double cost(const Eigen::MatrixXd& y, const Eigen::MatrixXd& p) override;
    
    size_t m_classes;
    std::vector<double> m_class_values;
};

#endif