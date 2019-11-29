#ifndef _ONEFORONE_H_
#define _ONEFORONE_H_

#include "classifier.h"
#include <core/parameters.h>
#include <vector>
#include <map>

class logistic_regression;

class one_for_one : public classifier
{
public:
    one_for_one();
    ~one_for_one();

    xt::xarray<double> predict(const xt::xarray<double>& x) override;

    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    
    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() override;
    
protected:
    std::vector<logistic_regression*> m_models;
};

#endif