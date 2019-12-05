#include "roc.h"

#include <xtensor/xview.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> roc::roc_curve(const xt::xarray<double>& y_proba, const xt::xarray<double>& y, size_t pos_label)
{
    xt::xarray<double> p_view = xt::reshape_view(y_proba, { y_proba.shape()[0]});
    xt::xarray<double> y_view = xt::reshape_view(y, { y.shape()[0]});

    xt::xarray<size_t> desc_p_indices = xt::argsort(p_view);
    xt::xarray<double> p_sort = xt::index_view(p_view, desc_p_indices);
    y_view = xt::index_view(y_view, desc_p_indices);
    xt::xarray<size_t> distinct_value_indices = xt::flatten_indices(xt::where(xt::diff(p_sort)));
    xt::xarray<size_t> threshold_idxs = xt::concatenate(xt::xtuple<>(distinct_value_indices, xt::xarray<size_t>({ y_view.size() - 1 })));

    xt::xarray<double> thresholds = xt::index_view(xt::cumsum(y_view), threshold_idxs);
    xt::xarray<double> tps = xt::index_view(xt::cumsum(y_view), threshold_idxs);
    xt::xarray<double> fps = 1 + threshold_idxs - tps;

    tps = xt::concatenate(xt::xtuple<>(xt::xarray<double>({ 0.0 }), tps));
    fps = xt::concatenate(xt::xtuple<>(xt::xarray<double>({ 0.0 }), fps));
    thresholds = xt::concatenate(xt::xtuple<>(xt::xarray<double>({ thresholds(0) + 1.0 }), thresholds));

    return std::make_tuple<>(fps / fps(fps.size() - 1), tps / tps(tps.size() - 1), thresholds);
}
