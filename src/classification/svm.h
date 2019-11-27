#ifndef _SVM_H_
#define _SVM_H_

#include "classifier.h"

class svm : public classifier
{
public:
    svm();
    svm(const svm&) = delete;
    ~svm() = default;

    xt::xarray<double> predict(const xt::xarray<double>& x) override;
    double score(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    void init_classes(size_t number_of_classes) override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() override;

private:
    xt::xarray<double> m_weights;
    size_t m_classes;
};

#endif