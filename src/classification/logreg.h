#ifndef _LOGREG_H_
#define _LOGREG_H_

#include "classifier.h"
#include <core/parameters.h>

class logistic_regression : public classifier
{
public:
    logistic_regression();
    logistic_regression(const logistic_regression&) = delete;
    ~logistic_regression() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void init_classes(size_t number_of_classes) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

protected:
    virtual double cost(const Eigen::MatrixXd& y, const Eigen::MatrixXd& p);
    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& p);

    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
    void calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);

    size_t m_classes;
    Eigen::MatrixXd m_weights;
};

#endif