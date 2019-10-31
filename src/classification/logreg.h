#ifndef _LOGREG_H_
#define _LOGREG_H_

#include <Eigen/Dense>

class logistic_regression
{
public:
    logistic_regression();
    ~logistic_regression() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x);
    double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);
    void train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations = 0);

    void set_weights(const Eigen::MatrixXd& weights);
    Eigen::MatrixXd weights();

protected:
    virtual double cost(const Eigen::MatrixXd& y, const Eigen::MatrixXd& p);
    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const Eigen::MatrixXd& p);

    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);
    void calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);

    double m_rate;
    double m_threshold;

    Eigen::MatrixXd m_weights;
};

#endif