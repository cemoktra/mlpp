#ifndef _ONEFORONE_H_
#define _ONEFORONE_H_

#include "model_interface.h"
#include <vector>
#include <map>

class logistic_regression;

class one_for_one : public model_interface
{
public:
    one_for_one();
    ~one_for_one();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;

    void train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations = 0) override;
    double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) override;
    
protected:
    std::vector<logistic_regression*> m_models;
    size_t m_classes;
};

#endif