#ifndef _MODEL_INTERFACE_H_
#define _MODEL_INTERFACE_H_

#include <Eigen/Dense>

class model_interface
{
public:
    model_interface() = default;
    ~model_interface() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) = 0;
    virtual void train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations = 0)  = 0;
    virtual double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) = 0;
};

#endif