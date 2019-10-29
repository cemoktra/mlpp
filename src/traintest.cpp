#include "traintest.h"

#include <random>
#include <numeric>

void test_train::split(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test, Eigen::MatrixXd& y_train, Eigen::MatrixXd& y_test, double test_proportion, bool shuffle)
{
    size_t test_size = y.rows() * test_proportion;
    size_t train_size = y.rows() - test_size;

    x_train.resize(train_size, x.cols());
    y_train.resize(train_size, 1);
    x_test.resize(test_size, x.cols());
    y_test.resize(test_size, 1);

    std::vector<size_t> indices (y.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 0.9999);

    for (auto i = 0; i < train_size; i++)
    {
        auto index = shuffle ? indices.begin() + floor(dis(gen) * indices.size()) : indices.begin();

        for (auto j = 0; j < x.cols(); j++) {
            x_train(i, j) = x(*index, j);
        }
        
        y_train(i, 0) = y(*index, 0);
        indices.erase(index);
    }

    for (auto i = 0; i < test_size; i++)
    {
        auto index = shuffle ? indices.begin() + floor(dis(gen) * indices.size()) : indices.begin();
        for (auto j = 0; j < x.cols(); j++) {
            x_test(i, j) = x(*index, j);
        }
        y_test(i, 0) = y(*index, 0);
        indices.erase(index);
    }
}