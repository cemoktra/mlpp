#ifndef _ONEFORONE_H_
#define _ONEFORONE_H_

#include <Eigen/Dense>
#include <vector>
#include <map>

class logistic_regression;

class one_for_one
{
public:
    one_for_one();
    ~one_for_one();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x);

    void train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations = 0);
    double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);
    
    void set_weights(const Eigen::MatrixXd& weights);
    Eigen::MatrixXd weights();
    
protected:
    std::vector<logistic_regression*> m_models;
    size_t m_classes;
};

#endif