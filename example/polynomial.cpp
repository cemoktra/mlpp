#include "../src/linreg.h"
#include "../src/polyfeatures.h"
#include "../src/traintest.h"
#include "../test/testdata.h"
#include <iostream>

int main(int argc, char** args)
{
    Eigen::MatrixXd data;
    std::cout << "reading data ... ";
    test_data::parse("diamonds.csv", {0, 4, 5, 6, 7, 8, 9}, data);
    std::cout << "done" << std::endl;

    Eigen::MatrixXd m_y(data.rows(), 1);
    m_y.col(0) = data.col(3);
    
    Eigen::MatrixXd m_x(data.rows(), 3);
    m_x.col(0) = data.col(4);
    m_x.col(1) = data.col(5);
    m_x.col(2) = data.col(6);
    
    polynomial_features pf (2, true);
    m_x = pf.transform(m_x);

    linear_regression lr;

    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    test_train::split(m_x, m_y, x_train, x_test, y_train, y_test);
    lr.train(x_train, y_train);

    std::cout << "weights: " << lr.weights().transpose() << std::endl;
    std::cout << "score: " << lr.score(x_test, y_test) << std::endl;
}