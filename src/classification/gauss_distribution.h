#ifndef _GAUSS_DISTRIBUTION_H_
#define _GAUSS_DISTRIBUTION_H_

#include "distribution.h"

class gauss_distribution : public distribution
{
public:
    gauss_distribution() = default;
    gauss_distribution(const gauss_distribution&) = delete;
    ~gauss_distribution() = default;

    void calc_weights(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    
    Eigen::MatrixXd weights() override;
    void set_weights(const Eigen::MatrixXd& weights) override;

private:
    Eigen::VectorXd calc_pfc(const Eigen::MatrixXd& x, size_t _class);

    Eigen::VectorXd m_pre_prop;
    Eigen::MatrixXd m_var;
    Eigen::MatrixXd m_mean;
};

#endif