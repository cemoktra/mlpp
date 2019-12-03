#ifndef _TRAINTEST_H_
#define _TRAINTEST_H_

#include <xtensor/xarray.hpp>
#include <stdexcept>

class train_test_split
{
public:
    train_test_split() = default;
    train_test_split(const train_test_split&) = delete;
    ~train_test_split() = default;

    void init(size_t rows, double test_proportion = 0.25, bool shuffle  = true);
    std::pair<xt::xarray<double>, xt::xarray<double>> split(const xt::xarray<double>& x);
    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> split(const xt::xarray<double>& x, const xt::xarray<double>& y);
    
    template<typename T>
    void split(const std::vector<T>& x, std::vector<T>& train, std::vector<T>& test) {
        train.clear();
        test.clear();
        for (auto i : m_train_indices)
            train.push_back(x[i]);
        for (auto i : m_test_indices)
            test.push_back(x[i]);
    }

    xt::xarray<int64_t> train_indices() const;
    xt::xarray<int64_t> test_indices() const;    

private:
    xt::xarray<int64_t> m_train_indices;
    xt::xarray<int64_t> m_test_indices;
};

static std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> do_train_test_split(const xt::xarray<double>& x, const xt::xarray<double>& y, double test_proportion = 0.25, bool shuffle  = true)
{
    train_test_split tt;
    tt.init(x.shape()[0], test_proportion, shuffle);
    return tt.split(x, y);
}

#endif