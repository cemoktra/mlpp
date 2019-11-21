#ifndef _BINOMIAL_DISTRIBUTION_H_
#define _BINOMIAL_DISTRIBUTION_H_

#include "distribution.h"

class binomial_distribution : public distribution
{
public:
    binomial_distribution() = default;
    binomial_distribution(const binomial_distribution&) = delete;
    ~binomial_distribution() = default;

    void calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    xt::xarray<double> predict(const xt::xarray<double>& x) override;
    
    xt::xarray<double> weights() override;
    void set_weights(const xt::xarray<double>& weights) override;

private:
    xt::xarray<double> m_pre_prop;
    xt::xarray<double> m_smooth;
    xt::xarray<double> m_feature_prop;
};

#endif