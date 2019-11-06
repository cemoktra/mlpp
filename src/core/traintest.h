#ifndef _TRAINTEST_H_
#define _TRAINTEST_H_

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

class train_test_split
{
public:
    train_test_split() = default;
    train_test_split(const train_test_split&) = delete;
    ~train_test_split() = default;

    void init(size_t rows, double test_proportion = 0.25, bool shuffle  = true);
    void split(const Eigen::MatrixXd& x, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test);
    void split(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test, Eigen::MatrixXd& y_train, Eigen::MatrixXd& y_test);
    
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

static void do_train_test_split(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, Eigen::MatrixXd& x_train, Eigen::MatrixXd& x_test, Eigen::MatrixXd& y_train, Eigen::MatrixXd& y_test, double test_proportion = 0.25, bool shuffle  = true)
{
    train_test_split tt;
    tt.init(x.rows(), test_proportion, shuffle);
    tt.split(x, y, x_train, x_test, y_train, y_test);
}

#endif