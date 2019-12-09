#ifndef _SCALER_H_
#define _SCALER_H_

#include <xtensor/xarray.hpp>

class scaler {
public:
    scaler() = default;
    scaler(const scaler&) = delete;
    ~scaler() = default;

    virtual void fit(const  xt::xarray<double>& x) = 0;
    virtual xt::xarray<double> transform(const  xt::xarray<double>& x) = 0;
    
    xt::xarray<double> fit_transform(const  xt::xarray<double>& x)
    {
        fit(x);
        return transform(x);
    }

    virtual xt::xarray<double> inverse_transform(const  xt::xarray<double>& x) = 0;
};

class standard_scaler : public scaler
{
public:
    standard_scaler() = default;
    standard_scaler(const standard_scaler&) = delete;
    ~standard_scaler() = default;

    void fit(const  xt::xarray<double>& x) override;
    xt::xarray<double> transform(const  xt::xarray<double>& x) override;

    xt::xarray<double> inverse_transform(const  xt::xarray<double>& x) override;

private:
    xt::xarray<double> m_mean;
    xt::xarray<double> m_stddev;
};

class normal_scaler : public scaler
{
public:
    normal_scaler() = default;
    normal_scaler(const normal_scaler&) = delete;
    ~normal_scaler() = default;

    void fit(const  xt::xarray<double>& x) override;
    xt::xarray<double> transform(const  xt::xarray<double>& x) override;

    xt::xarray<double> inverse_transform(const  xt::xarray<double>& x) override;

private:
    xt::xarray<double> m_mean;
    xt::xarray<double> m_scale;
};

#endif