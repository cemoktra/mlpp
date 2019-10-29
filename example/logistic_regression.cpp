#include "../src/logreg.h"
#include "../src/traintest.h"
#include "../src/normalize.h"
#include "../test/testdata.h"
#include <iostream>
#include <numeric>

int main(int argc, char** args)
{
    std::map<std::string, double> class_map;

    class_map["M"] = 1.0; // bad
    class_map["B"] = 0.0; // good

    std::vector<int> requested_cols (20);
    std::iota(requested_cols.begin(), requested_cols.end(), 1);

    Eigen::MatrixXd data;
    std::cout << "reading data ... ";
    test_data::parse("cancer.csv", requested_cols, data, class_map);
    std::cout << "done" << std::endl;

    Eigen::MatrixXd m_y(data.rows(), 1);
    m_y.col(0) = data.col(0);

    Eigen::MatrixXd m_x = data.block(0, 1, data.rows(), data.cols() - 1);
    m_x = normalize::transform(m_x);
    logistic_regression lr;

    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    test_train::split(m_x, m_y, x_train, x_test, y_train, y_test, 0.25);
    lr.train(x_train, y_train);

    std::cout << "weights: " << lr.weights().transpose() << std::endl;
    std::cout << "score: " << lr.score(x_test, y_test) << std::endl;
    
    return 0;
}
