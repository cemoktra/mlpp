#include "kfold.h"
#include <random>
#include <numeric>
#include <iostream>

kfold::kfold(size_t k, bool shuffle)
    : m_k(k)
    , m_shuffle(shuffle)
    , m_blockSize(0)
{
}

size_t kfold::k()
{
    return m_k;
}

void kfold::split(size_t index, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test, Eigen::MatrixXd& y_train, Eigen::MatrixXd& y_test)
{
    if (m_indices.size() != y.rows()) {
        prepareIndices(y.rows());
        m_blockSize = ceil(static_cast<double>(y.rows()) / m_k);
    }
    if (index >= m_k)
        throw std::out_of_range ("invalid split index");

    auto test_begin = std::next(m_indices.begin(), index * m_blockSize);
    auto test_end   = std::next(test_begin, m_blockSize - 1);
    test_end = std::min(test_end, m_indices.end());

    Eigen::MatrixXd _x_train(x.rows(), x.cols());
    Eigen::MatrixXd _y_train(y.rows(), y.cols());
    Eigen::MatrixXd _x_test(x.rows(), x.cols());
    Eigen::MatrixXd _y_test(y.rows(), y.cols());

    size_t train_idx = 0;
    size_t test_idx = 0;
    for (auto it = m_indices.begin(); it != m_indices.end(); ++it)
    {
        if (it < test_begin || it > test_end) {
            for (auto j = 0; j < x.cols(); j++)
                _x_train(train_idx, j) = x(*it, j);
            for (auto j = 0; j < y.cols(); j++)
                _y_train(train_idx, j) = y(*it, j);
            train_idx++;
        } else {
            for (auto j = 0; j < x.cols(); j++)
                _x_test(test_idx, j) = x(*it, j);
            for (auto j = 0; j < y.cols(); j++)
                _y_test(test_idx, j) = y(*it, j);
            test_idx++;
        }
    }

    x_train = _x_train.block(0, 0, train_idx, x.cols());
    y_train = _y_train.block(0, 0, train_idx, y.cols());
    x_test = _x_test.block(0, 0, test_idx, x.cols());
    y_test = _y_test.block(0, 0, test_idx, y.cols());
}

void kfold::prepareIndices(size_t count)
{
    m_indices.resize(count);
    std::iota(m_indices.begin(), m_indices.end(), 0);

    if (m_shuffle) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(m_indices.begin(), m_indices.end(), gen);
    }
}