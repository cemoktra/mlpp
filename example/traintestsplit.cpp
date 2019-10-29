#include "../src/linreg.h"
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
    
    Eigen::MatrixXd m_x(data.rows(), 2);
    m_x.col(0) = data.col(0);
    m_x.col(1) = Eigen::VectorXd::Ones(data.rows());
    
    linear_regression lr;

    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    test_train::split(m_x, m_y, x_train, x_test, y_train, y_test, 0.25, true);
    lr.train(x_train, y_train);

    std::cout << "weights: " << lr.weights().transpose() << std::endl;
    std::cout << "score: " << lr.score(x_test, y_test) << std::endl;
}
