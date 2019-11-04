#include "validation_curve.h"
#include "model.h"
#include "kfold.h"

Eigen::MatrixXd validation_curve::create(model *m, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::string& parameter, std::vector<double> parameter_values)
{
    kfold kf (4, true);
    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    Eigen::MatrixXd result (parameter_values.size(), 3);
    double test_score, train_score;
    size_t index = 0;

    for (auto parameter_value : parameter_values) {
        test_score = 0.0;
        train_score = 0.0;
        m->set_param(parameter, parameter_value);

        for (auto i = 0; i < kf.k(); ++i) {
            kf.split(i, x, y, x_train, x_test, y_train, y_test);
            m->train(x_train, y_train);
            train_score += m->score(x_train, y_train);
            test_score += m->score(x_test, y_test);
        }
        train_score /= kf.k();
        test_score /= kf.k();
        result(index, 0) = parameter_value;
        result(index, 1) = train_score;
        result(index, 2) = test_score;
        index++;
    }

    return result;
}