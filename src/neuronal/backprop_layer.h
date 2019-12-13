#ifndef _BACKPROP_LAYER_H_
#define _BACKPROP_LAYER_H_

#include <xtensor/xarray.hpp>

class dense_layer;

class backprop_layer {
public:
    backprop_layer(std::shared_ptr<dense_layer> layer);
    backprop_layer(const backprop_layer&) = delete;
    ~backprop_layer() = default;

    xt::xarray<double> execute(const xt::xarray<double>& g);

    inline const xt::xarray<double>& output_derivation() const { return m_dX; }
    inline const xt::xarray<double>& input_derivation() const { return m_dy; }

protected:
    std::shared_ptr<dense_layer> m_layer;
    xt::xarray<double> m_dX;
    xt::xarray<double> m_dy;
};

#endif