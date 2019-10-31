#ifndef _KFOLD_H_
#define _KFOLD_H_

#include <Eigen/Dense>

class kfold {
public:
    kfold(size_t k, bool shuffle = true);
    ~kfold() = default;

    void split(size_t index, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test, Eigen::MatrixXd& y_train, Eigen::MatrixXd& y_test);
    size_t k();

private:
    void prepareIndices(size_t count);

    size_t m_k;
    size_t m_blockSize;
    bool m_shuffle;
    std::vector<size_t> m_indices;
};

#endif