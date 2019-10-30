#include "../src/multinomial_logreg.h"
#include "../src/traintest.h"
#include "../src/normalize.h"
#include "../src/one_hot.h"
#include "../test/testdata.h"
#include <iostream>
#include <numeric>

int main(int argc, char** args)
{
    std::map<std::string, double> class_map;

    class_map["Apple"]  = 1.0;
    class_map["Orange"] = 2.0;
    class_map["Cola"]   = 3.0;

    std::vector<int> requested_cols (6);
    std::iota(requested_cols.begin(), requested_cols.end(), 3);

    Eigen::MatrixXd data;
    std::cout << "reading data ... ";
    test_data::parse("foods.csv", requested_cols, data, class_map);
    std::cout << "done" << std::endl;

    Eigen::MatrixXd m_y(data.rows(), 1);
    m_y.col(0) = data.col(5);
    m_y = one_hot::transform(m_y);

    Eigen::MatrixXd m_x = data.block(0, 0, data.rows(), data.cols() - 1);
    m_x = normalize::transform(m_x);

    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    test_train::split(m_x, m_y, x_train, x_test, y_train, y_test, 0.25, false);

    multinomial_logistic_regression mlr;
    mlr.train(x_train, y_train);
    std::cout << "score: " << mlr.score(x_test, y_test) << std::endl;
    
    return 0;
}
