#ifndef _BINOMIAL_DISTRIBUTION_H_
#define _BINOMIAL_DISTRIBUTION_H_

#include "distribution.h"

class binomial_distribution : public distribution
{
public:
    binomial_distribution() = default;
    binomial_distribution(const binomial_distribution&) = delete;
    ~binomial_distribution() = default;

    void calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    
    Eigen::MatrixXd weights() override;
    void set_weights(const Eigen::MatrixXd& weights) override;

private:
    Eigen::VectorXd m_pre_prop;
    Eigen::VectorXd m_smooth;
    Eigen::MatrixXd m_feature_prop;
};

#endif