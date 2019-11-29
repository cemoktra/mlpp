#ifndef _naive_bayes_H_
#define _naive_bayes_H_

#include "classifier.h"
#include <memory>

// TODO: add function for sparse matrices

class distribution;

class naive_bayes : public classifier
{
public:
    naive_bayes(std::shared_ptr<distribution> distribution);
    naive_bayes(const naive_bayes&) = delete;
    ~naive_bayes() = default;

    xt::xarray<double> predict(const xt::xarray<double>& x) override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() override;    

private:
    std::shared_ptr<distribution> m_distribution;

    xt::xarray<double> m_pre_prop;
    xt::xarray<double> m_var;
    xt::xarray<double> m_mean;
};

#endif