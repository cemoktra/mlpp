#ifndef _GAUSS_DISTRIBUTION_H_
#define _GAUSS_DISTRIBUTION_H_

#include "distribution.h"

class gauss_distribution : public distribution
{
public:
    gauss_distribution() = default;
    gauss_distribution(const gauss_distribution&) = delete;
    ~gauss_distribution() = default;

    void calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    xt::xarray<double> predict(const xt::xarray<double>& x) override;
    
    xt::xarray<double> weights() override;
    void set_weights(const xt::xarray<double>& weights) override;

private:
    double m_epsilon;
    xt::xarray<double> m_class_prior;
    xt::xarray<double> m_theta;
    xt::xarray<double> m_sigma;
};

#endif