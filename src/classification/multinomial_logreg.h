#ifndef _MULTINOMIALLOGREG_H_
#define _MULTINOMIALLOGREG_H_

#include "logreg.h"
#include <functional>

class multinomial_logistic_regression : public logistic_regression
{
public:
    multinomial_logistic_regression();
    ~multinomial_logistic_regression() = default;

    xt::xarray<double> predict(const xt::xarray<double>& x) const override;

protected:
    xt::xarray<double> softmax(const xt::xarray<double>& z) const;

    double cost(const xt::xarray<double>& y, const xt::xarray<double>& p) const override;
    
    size_t m_classes;
    std::vector<double> m_class_values;
};

#endif