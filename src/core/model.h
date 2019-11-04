#ifndef _MODEL_H_
#define _MODEL_H_

#include "parameters.h"

class model : public parameters {
public:
    model() = default;
    model(const model&) = delete;
    ~model() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) = 0;
    virtual double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) = 0;
    virtual void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) = 0;

    virtual void set_weights(const Eigen::MatrixXd& weights) = 0;
    virtual Eigen::MatrixXd weights() = 0;
};

#endif