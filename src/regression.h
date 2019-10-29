#ifndef _REGRESSION_H_
#define _REGRESSION_H_

#include <Eigen/Dense>
#include <vector>

class regression
{
public:
    regression(double rate, double threshold);
    ~regression() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) = 0;

    void train(const Eigen::MatrixXd& x, Eigen::MatrixXd& y, size_t maxIterations = 0);
    virtual double score(const Eigen::MatrixXd& x, Eigen::MatrixXd& y);

    void set_weights(const Eigen::MatrixXd& weights);
    Eigen::MatrixXd weights();
    
protected:
    virtual Eigen::MatrixXd calc_weights(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) = 0;

    double m_rate;
    double m_threshold;

    Eigen::MatrixXd m_weights;
    Eigen::MatrixXd m_weight_updates;
};

#endif