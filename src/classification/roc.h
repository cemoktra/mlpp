#ifndef _ROC_H_
#define _ROC_H_

#include <xtensor/xarray.hpp>

class roc {
public:
    roc() = default;
    roc(const roc&) = delete;
    ~roc() = default;

    static std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> roc_curve(const xt::xarray<double>& y_proba, const xt::xarray<double>& y, size_t pos_label = 1);
};

#endif