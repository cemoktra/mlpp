#include "traintest.h"

#include <numeric>
#include <random>

void test_train::split(const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::vector<std::vector<double>>& x_train, std::vector<std::vector<double>>& x_test, std::vector<double>& y_train, std::vector<double>& y_test, double test_proportion)
{
    size_t test_size = y.size() * test_proportion;
    size_t train_size = y.size() - test_size;

    x_train.resize(x.size());    
    x_test.resize(x.size());

    std::vector<size_t> indices (y.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 0.9999);

    for (auto i = 0; i < train_size; i++)
    {
        auto index = indices.begin() + floor(dis(gen) * indices.size());
        for (auto j = 0; j < x.size(); j++) {
            x_train[j].push_back(x[j][*index]);
        }
        
        y_train.push_back(y[*index]);
        indices.erase(index);
    }

    for (auto i = 0; i < test_size; i++)
    {
        auto index = indices.begin() + floor(dis(gen) * indices.size());
        for (auto j = 0; j < x.size(); j++) {
            x_test[j].push_back(x[j][*index]);
        }
        y_test.push_back(y[*index]);
        indices.erase(index);
    }
}