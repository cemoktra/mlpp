#ifndef _LOGREG_H_
#define _LOGREG_H_

#include "regression.h"

class logistic_regression : public regression
{
public:
    logistic_regression();
    ~logistic_regression() = default;

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) override;
    
protected:
    virtual double cost(const Eigen::MatrixXd& y, const Eigen::MatrixXd& p);
    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& p);

    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
    
    Eigen::MatrixXd calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) override;
};

#endif