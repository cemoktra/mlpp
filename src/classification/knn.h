#ifndef _KNN_H_
#define _KNN_H_

#include "classifier.h"

class knn : public classifier
{
public:
    knn(size_t k);
    knn(const knn&) = delete;
    ~knn();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::string>& classes, size_t maxIterations = 0) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

private:
    Eigen::MatrixXd m_x_train;
    Eigen::MatrixXd m_y_train;
    size_t m_k;
    size_t m_classes;
    double *m_kdistances;
    size_t *m_knearest_classes;
    size_t *m_class_count;
};

#endif