#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include "classifier.h"
#include <core/parameters.h>

class decision_tree;

class random_forest : public classifier
{
public:
    random_forest();
    ~random_forest();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void init_classes(const std::vector<std::string>& classes) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

    void set_param(const std::string& name, double new_value) override;

private:
    decision_tree **m_trees;
    std::vector<std::string> m_classes;
    size_t m_class_count;
};

#endif