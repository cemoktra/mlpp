#include "kmeans.h"

#include <xtensor/xrandom.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>


kmeans::kmeans()
{
    register_param("k", 5.0);
    register_param("threshold", 0.0001);
}

void kmeans::fit(const xt::xarray<double>& X)
{
    size_t k = static_cast<size_t>(get_param("k"));
    std::random_device rd;
    std::mt19937 gen(rd());


    // initialize centers
    // TODO: option to init centers
    auto minmax = xt::eval(xt::minmax(X))(0);
    m_centers = xt::random::rand<double>(std::vector<size_t>({k, X.shape()[1]}), minmax[0], minmax[1], gen);
    m_labels = xt::zeros<size_t>({X.shape()[0]});
    m_inertia = xt::zeros<double>({X.shape()[1]});

    while (true) {
        // label samples to closest centroid
        for (auto sample = 0; sample < X.shape()[0]; sample++) {
            auto distances = xt::sqrt(xt::sum(xt::square(m_centers - xt::view(X, xt::range(sample, sample + 1), xt::all())), {1}));
            m_labels(sample) = xt::argmin(distances)(0);
        }

        // move centroids
        double delta = 0.0;
        for (auto label = 0; label < X.shape()[1]; label++) {
            auto indices = xt::flatten_indices(xt::argwhere(xt::equal(m_labels, label)));
            auto label_data = xt::view(X, xt::keep(indices), xt::all());
            auto new_center = xt::sum(label_data, {0}) / label_data.shape()[0];
            auto distance = xt::sqrt(xt::sum(xt::square(new_center - xt::view(m_centers, xt::range(label, label + 1), xt::all())), {1}))(0);
            auto distances = xt::sqrt(xt::sum(xt::square(label_data - new_center), {1}));

            delta += distance;            
            xt::view(m_centers, xt::range(label, label + 1), xt::all()) = new_center;
            m_inertia(label) = xt::sum(distances)(0);
        }
        delta /= m_labels.shape()[0];

        // break
        if (delta < get_param("threshold"))
            break;
    }
}