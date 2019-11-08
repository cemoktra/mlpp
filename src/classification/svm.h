#ifndef _SVM_H_
#define _SVM_H_

#include "classifier.h"

class svm : public classifier
{
public:
    svm();
    svm(const svm&) = delete;
    ~svm() = default;

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void init_classes(size_t number_of_classes) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

private:
    Eigen::MatrixXd m_weights;
};

#endif