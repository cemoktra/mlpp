#ifndef _ONEFORONE_H_
#define _ONEFORONE_H_

#include "classifier.h"
#include <core/parameters.h>
#include <vector>
#include <map>

class logistic_regression;

class one_for_one : public classifier
{
public:
    one_for_one();
    ~one_for_one();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;

    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void init_classes(const std::vector<std::string>& classes) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    
    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;
    
protected:
    std::vector<logistic_regression*> m_models;
    std::vector<std::string> m_classes;
    size_t m_class_count;
};

#endif