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

    xt::xarray<double> predict(const xt::xarray<double>& x) override;
    double score(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    void init_classes(size_t number_of_classes) override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() override;

private:
    xt::xarray<double> m_x_train;
    xt::xarray<double> m_y_train;
    xt::xarray<double> m_kdistances;
    xt::xarray<size_t> m_knearest_classes;
    xt::xarray<size_t> m_class_count;

    size_t m_classes;
    // double *m_kdistances;
    // size_t *m_knearest_classes;
    // size_t *m_class_count;
};

#endif