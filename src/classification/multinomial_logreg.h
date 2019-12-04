#ifndef _MULTINOMIALLOGREG_H_
#define _MULTINOMIALLOGREG_H_

#include "logreg.h"

class multinomial_logistic_regression : public logistic_regression
{
public:
    multinomial_logistic_regression();
    ~multinomial_logistic_regression() = default;

protected:
    xt::xarray<double> activation(const xt::xarray<double>& x) const override;
    xt::xarray<double> reverse_activation(const xt::xarray<double>& y) const override;
};

#endif