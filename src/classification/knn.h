#ifndef _KNN_H_
#define _KNN_H_

#include "classifier.h"
#include <core/parameters.h>

class knn : public classifier
{
public:
    knn();
    knn(const knn&) = delete;
    ~knn() = default;

    xt::xarray<double> predict(const xt::xarray<double>& x) const override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() const override;

private:
    xt::xarray<double> m_x_train;
    xt::xarray<double> m_y_train;
};

#endif