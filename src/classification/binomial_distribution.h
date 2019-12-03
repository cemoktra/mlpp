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
    xt::xarray<double> predict(const xt::xarray<double>& x) const override;
    
    xt::xarray<double> weights() const override;
    void set_weights(const xt::xarray<double>& weights) override;

private:
    void update_feature_log_prior(const xt::xarray<double>& feature_count);
    void update_class_log_prior();

    xt::xarray<double> m_class_prior;
    xt::xarray<double> m_feature_log_prob;
    xt::xarray<double> m_class_log_prior;
};

#endif