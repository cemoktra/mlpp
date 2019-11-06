#include "learning_curve.h"
#include "model.h"
#include "traintest.h"

Eigen::MatrixXd learning_curve::create(model *m, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    Eigen::MatrixXd result (9, 3);
    double test_score, train_score;
    size_t index = 0;

    for (auto split = 0.9; fabs(split) > 0.001; split -= 0.1) {
        test_score = 0.0;
        train_score = 0.0;
        for (auto i = 0; i < 3; i++) {
            do_train_test_split(x, y, x_train, x_test, y_train, y_test, split, true);
            m->train(x_train, y_train);
            train_score += m->score(x_train, y_train);
            test_score += m->score(x_test, y_test);
        }
        train_score /= 3.0;
        test_score /= 3.0;
        result(index, 0) = split;
        result(index, 1) = train_score;
        result(index, 2) = test_score;
        index++;
    }

    return result;
}