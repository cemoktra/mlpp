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

    virtual xt::xarray<double> predict(const xt::xarray<double>& x) override;
    double score(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    void init_classes(size_t number_of_classes) override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() override;

protected:
    virtual double cost(const xt::xarray<double>& y, const xt::xarray<double>& p);
    virtual xt::xarray<double> gradient(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<double>& p);

    xt::xarray<double> sigmoid(const xt::xarray<double>& x);
    void calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y);

    size_t m_classes;
    xt::xarray<double> m_weights;
};

#endif