#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <Eigen/Dense>

class classifier {
public:
    classifier() = default;
    classifier(const classifier&) = delete;
    ~classifier() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) = 0;
    virtual double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) = 0;
    virtual void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::string>& classes, size_t maxIterations = 0) = 0;

    virtual void set_weights(const Eigen::MatrixXd& weights) = 0;
    virtual Eigen::MatrixXd weights() = 0;
};

#endif