#include "../src/linreg.h"
#include "../src/kfold.h"
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

    Eigen::MatrixXd x_train, x_test, y_train, y_test, mean_weights;

    kfold kf(4, false);
    linear_regression lr;

    for (auto i = 0; i < kf.k(); ++i) {
        kf.split(i, m_x, m_y, x_train, x_test, y_train, y_test);
        lr.train(x_train, y_train);
        
        std::cout << "KFOLD " << i + 1 << std::endl;
        std::cout << "coeffs: ";
        std::cout << lr.weights() << std::endl;
        std::cout << "score: " << lr.score(x_test, y_test) << std::endl;

        if (i > 0)
            mean_weights += lr.weights();
        else 
            mean_weights = lr.weights();
    }
    mean_weights /= kf.k();

    lr.set_weights(mean_weights);
    std::cout << "mean coeffs: ";
    std::cout << lr.weights() << std::endl;
    std::cout << "mean score: " << lr.score(x_test, y_test) << std::endl;
}
