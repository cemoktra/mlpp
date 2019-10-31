#ifndef _LINREG_H_
#define _LINREG_H_

#include "Eigen/Dense"

class linear_regression
{
public:
    linear_regression();
    ~linear_regression() = default;

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x);

    void train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations = 0);
    virtual double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);

    void set_weights(const Eigen::MatrixXd& weights);
    Eigen::MatrixXd weights();

private:
    void calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);

    double m_rate;
    double m_threshold;

    Eigen::MatrixXd m_weights;
};

#endif