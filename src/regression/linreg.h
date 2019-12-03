#ifndef _LINREG_H_
#define _LINREG_H_

#include <core/model.h>
#include <core/parameters.h>

class linear_regression : public model
{
public:
    linear_regression();
    ~linear_regression() = default;

    xt::xarray<double> predict(const xt::xarray<double>& x) const override;

    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    double score(const xt::xarray<double>& x, const xt::xarray<double>& y) const override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() const override;

private:
    void calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y);

    xt::xarray<double> m_weights;
};

#endif