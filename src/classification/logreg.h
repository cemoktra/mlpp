#ifndef _LOGREG_H_
#define _LOGREG_H_

#include "classifier.h"
#include <core/parameters.h>

class logistic_regression : public classifier
{
public:
    logistic_regression();
    logistic_regression(const logistic_regression&) = delete;
    ~logistic_regression() = default;

    virtual xt::xarray<double> predict(const xt::xarray<double>& x) const override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
  
    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() const override;

protected:
    void calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y);

    xt::xarray<double> m_weights;

    virtual xt::xarray<double> activation(const xt::xarray<double>& x) const;
    virtual xt::xarray<double> reverse_activation(const xt::xarray<double>& y) const;
};

#endif