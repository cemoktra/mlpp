#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <core/model.h>
#include <xtensor/xsort.hpp>

class classifier : public model {
public:
    classifier() = default;
    classifier(const classifier&) = delete;
    ~classifier() = default;
    
    virtual xt::xarray<double> predict_class(const xt::xarray<double>& x) const
    {
        auto proba = predict(x);
        if (proba.shape()[1] > 1)
            return xt::argmax(proba, {1});
        else
            return xt::where(proba > 0.5, 1.0, 0-0);
    }

    void init_classes(size_t number_of_classes) { m_classes = number_of_classes; }

    double score(const xt::xarray<double>& x, const xt::xarray<double>& y) const override
    {
        return score(confusion(x, y));
    }

    double score(const xt::xarray<double>& confusion_matrix) const
    {
        return xt::sum(xt::diagonal(confusion_matrix))[0];
    }

    xt::xarray<double> confusion(const xt::xarray<double>& x, const xt::xarray<double>& y) const
    {
        xt::xarray<double> confusion = xt::zeros<double>(std::vector<size_t>({m_classes, m_classes}));
        xt::xarray<size_t> p_class = xt::cast<size_t>(predict_class(x));
        xt::xarray<size_t> t_class;

        if (y.shape().size() > 1 && y.shape()[1] > 1)
            t_class = xt::argmax(y, {1});
        else
            t_class = y;
        t_class.reshape(p_class.shape());

        auto t_it = t_class.begin();
        for (auto p_val : p_class) {
            confusion(p_val, *t_it) += 1.0;
            t_it++;
        }
        return confusion / t_class.shape()[0];
    }

protected:
    size_t m_classes;
};

#endif