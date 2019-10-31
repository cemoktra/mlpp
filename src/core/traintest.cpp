#include "traintest.h"

#include <random>
#include <numeric>

void test_train::split(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test, Eigen::MatrixXd& y_train, Eigen::MatrixXd& y_test, double test_proportion, bool shuffle)
{
    size_t test_size = y.rows() * test_proportion;
    size_t train_size = y.rows() - test_size;

    x_train.resize(train_size, x.cols());
    y_train.resize(train_size, y.cols());
    x_test.resize(test_size, x.cols());
    y_test.resize(test_size, y.cols());

    std::vector<size_t> indices (x.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 0.9999);

    for (auto i = 0; i < train_size; i++)
    {
        auto index = shuffle ? indices.begin() + floor(dis(gen) * indices.size()) : indices.begin();

        for (auto j = 0; j < x.cols(); j++)
            x_train(i, j) = x(*index, j);
        for (auto j = 0; j < y.cols(); j++)
            y_train(i, j) = y(*index, j);
        indices.erase(index);
    }

    for (auto i = 0; i < test_size; i++)
    {
        auto index = shuffle ? indices.begin() + floor(dis(gen) * indices.size()) : indices.begin();
        for (auto j = 0; j < x.cols(); j++)
            x_test(i, j) = x(*index, j);
        for (auto j = 0; j < y.cols(); j++)
            y_test(i, j) = y(*index, j);
        indices.erase(index);
    }
}