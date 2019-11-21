#ifndef _TRAINTEST_H_
#define _TRAINTEST_H_

#include <xtensor/xarray.hpp>
#include <vector>
#include <stdexcept>

class train_test_split
{
public:
    train_test_split() = default;
    train_test_split(const train_test_split&) = delete;
    ~train_test_split() = default;

    void init(size_t rows, double test_proportion = 0.25, bool shuffle  = true);
    void split(const xt::xarray<double>& x, xt::xarray<double>& x_train, xt::xarray<double>& x_test);
    void split(const xt::xarray<double>& x, const xt::xarray<double>& y, xt::xarray<double>& x_train, xt::xarray<double>& x_test, xt::xarray<double>& y_train, xt::xarray<double>& y_test);
    
    template<typename T>
    void split(const std::vector<T>& x, std::vector<T>& train, std::vector<T>& test) {
        if (x.size() != m_train_indices.size() + m_test_indices.size())
            throw std::invalid_argument("vector x size does not match initialized size");
        train.clear();
        test.clear();
        for (auto i : m_train_indices)
            train.push_back(x[i]);
        for (auto i : m_test_indices)
            test.push_back(x[i]);
    }

    std::vector<size_t> train_indices() const;
    std::vector<size_t> test_indices() const;    

private:
    std::vector<size_t> m_train_indices;
    std::vector<size_t> m_test_indices;
};

static void do_train_test_split(const xt::xarray<double>& x, const xt::xarray<double>& y, xt::xarray<double>& x_train, xt::xarray<double>& x_test, xt::xarray<double>& y_train, xt::xarray<double>& y_test, double test_proportion = 0.25, bool shuffle  = true)
{
    train_test_split tt;
    tt.init(x.shape()[0], test_proportion, shuffle);
    tt.split(x, y, x_train, x_test, y_train, y_test);
}

#endif