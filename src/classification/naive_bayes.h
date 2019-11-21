#ifndef _naive_bayes_H_
#define _naive_bayes_H_

#include "classifier.h"
#include <memory>

// TODO: add function for sparse matrices

class distribution;

class naive_bayes : public classifier
{
public:
    naive_bayes(std::shared_ptr<distribution> distribution);
    naive_bayes(const naive_bayes&) = delete;
    ~naive_bayes() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void init_classes(size_t number_of_classes) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;    

private:
    size_t m_number_of_classes;
    std::shared_ptr<distribution> m_distribution;
};

#endif