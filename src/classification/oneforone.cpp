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

            for (auto k = 0; k < x.shape()[0]; k++) {
                if (p(k, 0) > 0.5)
                    prediction_count(k, i) += 1.0;
                else
                    prediction_count(k, j) += 1.0;
            }

            model++;
        }
    }

    xt::xarray<double> prediction_result = xt::zeros<double>({ x.shape()[0], size_t(2) });
    xt::view(prediction_result, xt::all(), xt::range(1, 2)) = xt::zeros<double>( { x.shape()[0], size_t(1) }) - 1;

    for (auto k = 0; k < x.shape()[0]; k++) {
        for (auto i = 0; i < m_number_of_classes; i++) {
            if (prediction_count(k, i) > prediction_result(k, 0))
            {
                prediction_result(k, 0) = prediction_count(k, i);
                prediction_result(k, 1) = i;
            }
        }
    }
    return prediction_result;
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
            bool first = true;
            for (auto k = 0; k < y.shape()[0]; k++)
            {
                bool include_data = y.shape()[1] > 1 ? 
                    y(k, i) > 0.0 || y(k, j) > 0.0 :
                    y(k, 0) == i || y(k, 0) == j;
                if (include_data)
                {                
                    if (!first) {
                        xt::xarray<double> new_x = xt::zeros<double>({ x_.shape()[0] + 1, x_.shape()[1]});
                        xt::xarray<double> new_y = xt::zeros<double>({ y_.shape()[0] + 1, y_.shape()[1]});
                        
                        xt::view(new_x, xt::range(0, x_.shape()[0], xt::all())) = x_;
                        xt::view(new_y, xt::range(0, y_.shape()[0], xt::all())) = y_;

                        xt::view(new_x, xt::range(x_.shape()[1], xt::placeholders::_), xt::all()) = xt::view(x, xt::range(k, k + 1), xt::all());
                        xt::view(new_y, xt::range(y_.shape()[1], xt::placeholders::_), xt::all()) = xt::view(y, xt::range(k, k + 1), xt::all());

                        x_ = new_x;
                        y_ = new_y;
                    } else {
                        first = false;
                        x_ = xt::view(x, xt::range(k, k + 1), xt::all());
                        y_ = xt::view(y, xt::range(k, k + 1), xt::all());
                    }
                }
            }
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
    xt::xarray<size_t> predict_class = xt::argmax(p, {1});
    xt::xarray<size_t> target_class;
    
    if (y.shape()[1] > 1)
        target_class = xt::argmax(p, {1});
    else
        target_class = y;
    target_class.reshape(predict_class.shape());
    return xt::sum(xt::equal(predict_class, target_class))(0) / static_cast<double>(y.shape()[0]);
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