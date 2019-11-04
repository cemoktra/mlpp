#ifndef _LINREG_H_
#define _LINREG_H_

#include "Eigen/Dense"
#include <core/model.h>
#include <core/parameters.h>

class linear_regression : public model
{
public:
    linear_regression();
    ~linear_regression() = default;

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;

    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

private:
    void calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);

    Eigen::MatrixXd m_weights;
};

#endif