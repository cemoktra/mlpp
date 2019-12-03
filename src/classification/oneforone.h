#ifndef _ONEFORONE_H_
#define _ONEFORONE_H_

#include "classifier.h"
#include <core/parameters.h>
#include <vector>
#include <xtensor/xview.hpp>

template<class C>
class one_for_one : public classifier
{
public:
    one_for_one();
    ~one_for_one();

    xt::xarray<double> predict(const xt::xarray<double>& x) const override;

    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    
    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() const override;
    
protected:
    std::vector<C*> m_models;
};

template<class C>
one_for_one<C>::one_for_one()
{
    C impl;
    copy_params(impl);
}

template<class C>
one_for_one<C>::~one_for_one()
{
    for (auto lr : m_models)
        delete lr;
}

template<class C>
xt::xarray<double> one_for_one<C>::predict(const xt::xarray<double>& x) const
{
    xt::xarray<double> common_probability = xt::zeros<double>({ x.shape()[0], m_classes });
    size_t model = 0;
    for (auto i = 0; i < m_classes - 1; i++) {
        auto p1 = xt::view(common_probability, xt::all(), xt::range(i, i + 1));

        for (auto j = i + 1; j < m_classes; j++) {
            auto p2 = xt::view(common_probability, xt::all(), xt::range(j, j + 1));
            auto p = m_models[model++]->predict(x);
            p1 += xt::view(p, xt::all(), xt::range(0, 1));
            p2 += xt::view(p, xt::all(), xt::range(1, 2));
        }
    }
    return common_probability / static_cast<double>(m_models.size());
}

template<class C>
void one_for_one<C>::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    for (auto lr : m_models)
        delete lr;

    for (auto i = 0; i < m_classes - 1; i++) {
        for (auto j = i + 1; j < m_classes; j++) {
            xt::xarray<double> y_, x_;

            auto idx = (y.shape()[1] > 1) ? 
                xt::flatten_indices(xt::argwhere(xt::view(y, xt::all(), xt::range(i, i + 1)) > 0.0 || xt::view(y, xt::all(), xt::range(j, j + 1)) > 0.0)) :
                xt::flatten_indices(xt::argwhere(xt::equal(y, i) || xt::equal(y, j)));
            x_ = xt::view(x, xt::keep(idx), xt::all());
            y_ = xt::view(y, xt::keep(idx), xt::all());

            y_ = xt::where(y_ > 0.0, 1, 0);
            C *impl = new C();
            impl->copy_params(*this);           
            impl->init_classes(m_classes);
            impl->train(x_, y_);
            m_models.push_back(impl);
        }
    }
}

template<class C>
void one_for_one<C>::set_weights(const xt::xarray<double>& weights)
{
    // TODO: ensure models are created
    for (auto i = 0; i < weights.shape()[1]; i++)
        m_models[i]->set_weights(xt::view(weights, xt::all(), xt::range(i, i + 1)));
}

template<class C>
xt::xarray<double> one_for_one<C>::weights() const
{
    xt::xarray<double> weights;
    for (auto i = 0; i < m_models.size(); i++)
        weights = i == 0 ? m_models[i]->weights() : xt::concatenate(std::make_tuple<>(weights, m_models[i]->weights()), 1);
    return weights;
}

#endif