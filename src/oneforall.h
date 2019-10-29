#ifndef _ONEFORALL_H_
#define _ONEFORALL_H_

#include "model_interface.h"
#include <vector>
#include <map>

class logistic_regression;

class one_for_all : public model_interface
{
public:
    one_for_all(const std::map<std::string, double>& class_map);
    ~one_for_all();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;

    void train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations = 0) override;
    double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) override;
    
    std::string mapped_class(double value);

protected:
    std::vector<logistic_regression*> m_models;
    std::map<std::string, double> m_class_map;
    size_t m_classes;
    std::vector<double> m_class_values;
};

#endif