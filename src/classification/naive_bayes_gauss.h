#ifndef _NAIVE_BAYES_GAUSS_H_
#define _NAIVE_BAYES_GAUSS_H_

#include "classifier.h"

class naive_bayes_gauss : public classifier
{
public:
    naive_bayes_gauss();
    naive_bayes_gauss(const naive_bayes_gauss&) = delete;
    ~naive_bayes_gauss() = default;

    virtual Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void init_classes(const std::vector<std::string>& classes) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;    

private:
    Eigen::VectorXd mean(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, size_t _class);
    Eigen::VectorXd variance(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, size_t _class);
    Eigen::VectorXd pfc(const Eigen::MatrixXd& x, size_t _class);
    Eigen::MatrixXd pinv(const Eigen::MatrixXd& x);

    std::vector<std::string> m_classes;

    Eigen::VectorXd m_pre_prop;
    Eigen::MatrixXd m_var;
    Eigen::MatrixXd m_mean;
};

#endif