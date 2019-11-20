#include "oneforone.h"
#include "logreg.h"
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>

one_for_one::one_for_one() 
    : classifier()
{
    register_param("learning_rate", 0.02);
    register_param("threshold", 0.0001);
    register_param("max_iterations", 0);
}

one_for_one::~one_for_one()
{
    for (auto lr : m_models)
        delete lr;
}

xt::xarray<double> one_for_one::predict(const xt::xarray<double>& x)
{
    xt::xarray<double> prediction_count = xt::zeros<double>({ x.shape()[0], m_number_of_classes });

    size_t model = 0;
    for (auto i = 0; i < m_number_of_classes - 1; i++) {
        for (auto j = i + 1; j < m_number_of_classes; j++) {
            auto p = m_models[model]->predict(x);
            xt::xarray<size_t> target_class = xt::argmax(p, {1});
            auto idx1 = xt::flatten_indices(xt::argwhere(target_class < 1.0));
            auto idx2 = xt::flatten_indices(xt::argwhere(target_class > 0.0));
            auto p1 = xt::view(prediction_count, xt::keep(idx1), xt::range(i, i + 1));
            auto p2 = xt::view(prediction_count, xt::keep(idx2), xt::range(j, j + 1));
            p1 += 1.0;
            p2 += 1.0;
            model++;
        }
    }

    return xt::argmax(prediction_count, {1});
}

void one_for_one::init_classes(size_t number_of_classes)
{
    m_number_of_classes = number_of_classes;
}

void one_for_one::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    for (auto lr : m_models)
        delete lr;

    for (auto i = 0; i < m_number_of_classes - 1; i++) {
        for (auto j = i + 1; j < m_number_of_classes; j++) {
            xt::xarray<double> y_, x_;

            auto idx = (y.shape()[1] > 1) ? 
                xt::flatten_indices(xt::argwhere(xt::view(y, xt::all(), xt::range(i, i + 1)) > 0.0 || xt::view(y, xt::all(), xt::range(j, j + 1)) > 0.0)) :
                xt::flatten_indices(xt::argwhere(xt::equal(y, i) || xt::equal(y, j)));
            x_ = xt::view(x, xt::keep(idx), xt::all());
            y_ = xt::view(y, xt::keep(idx), xt::all());

            y_ = xt::where(y_ > 0.0, 1, 0);
            logistic_regression *lr = new logistic_regression();
            lr->set_param("learning_rate", get_param("learning_rate"));
            lr->set_param("threshold", get_param("threshold"));
            lr->set_param("max_iterations", get_param("max_iterations"));
            m_models.push_back(lr);
            lr->init_classes(m_number_of_classes);
            lr->train(x_, y_);
        }
    }
}

double one_for_one::score(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> p = predict(x);
    xt::xarray<size_t> target_class;
    
    if (y.shape()[1] > 1)
        target_class = xt::argmax(p, {1});
    else
        target_class = y;
    target_class.reshape(p.shape());
    return xt::sum(xt::equal(p, target_class))(0) / static_cast<double>(y.shape()[0]);
}

void one_for_one::set_weights(const xt::xarray<double>& weights)
{
    for (auto i = 0; i < weights.shape()[1]; i++)
        m_models[i]->set_weights(xt::view(weights, xt::all(), xt::range(i, i + 1)));
}

xt::xarray<double> one_for_one::weights()
{
    xt::xarray<double> weights;

    for (auto i = 0; i < m_models.size(); i++) {
        if (i == 0)
            weights = m_models[i]->weights();
        else {
            xt::xarray<double> new_weights = xt::zeros<double>({ weights.shape()[0], weights.shape()[1] + 1});
            xt::view(new_weights, xt::all(), xt::range(0, weights.shape()[0])) = weights;
            xt::view(new_weights, xt::all(), xt::range(weights.shape()[1], xt::placeholders::_)) = xt::view(m_models[i]->weights(), xt::all(), xt::range(0, 1));
            weights = new_weights;
        }
    }

    return weights;
}