#include <regression/linreg.h>
#include <core/traintest.h>
#include <core/kfold.h>
#include <core/testdata.h>
#include <core/polyfeatures.h>
#include <iostream>

int main(int argc, char** args)
{
    Eigen::MatrixXd data;
    linear_regression lr;
    bool shuffle = true;
    double split = 0.25;

    // parse csv
    std::cout << "reading data ... ";
    test_data::parse("diamonds.csv", {0, 4, 5, 6, 7, 8, 9}, data);
    std::cout << "done" << std::endl;

    // prepare y matrix to price
    Eigen::MatrixXd m_y(data.rows(), 1);
    m_y.col(0) = data.col(3);
    
    // prepare x matrix to use carat
    Eigen::MatrixXd m_x(data.rows(), 2);
    m_x.col(0) = data.col(0);
    m_x.col(1) = Eigen::VectorXd::Ones(data.rows());
    
    // regression train-test-split
    {
        std::cout << "regression with one feature and train-test-split:" << std::endl;

        Eigen::MatrixXd x_train, x_test, y_train, y_test;
        test_train::split(m_x, m_y, x_train, x_test, y_train, y_test, split, shuffle);
        lr.train(x_train, y_train);
        std::cout << "  weights: " << lr.weights().transpose() << std::endl;
        std::cout << "  score: " << lr.score(x_test, y_test) << std::endl;
    }

    // regression kfold
    {
        std::cout << "regression with one feature kfold:" << std::endl;

        Eigen::MatrixXd x_train, x_test, y_train, y_test, mean_weights;
        kfold kf(4, shuffle);

        for (auto i = 0; i < kf.k(); ++i) {
            kf.split(i, m_x, m_y, x_train, x_test, y_train, y_test);
            lr.train(x_train, y_train);
            
            std::cout << "  fold " << i + 1 << std::endl;
            std::cout << "    coeffs: " << lr.weights().transpose() << std::endl;
            std::cout << "    score: " << lr.score(x_test, y_test) << std::endl;

            if (i > 0)
                mean_weights += lr.weights();
            else 
                mean_weights = lr.weights();
        }
        mean_weights /= kf.k();

        lr.set_weights(mean_weights);
        std::cout << "  mean coeffs: ";
        std::cout << lr.weights().transpose() << std::endl;
        std::cout << "  mean score: " << lr.score(x_test, y_test) << std::endl;
    }

    // regression with multi vars
    m_x.resize(data.rows(), 4);
    m_x.col(0) = data.col(4);
    m_x.col(1) = data.col(5);
    m_x.col(2) = data.col(6);
    m_x.col(3) = Eigen::VectorXd::Ones(data.rows());    
    
    {
        std::cout << "regression with multiple features and train-test-split:" << std::endl;

        Eigen::MatrixXd x_train, x_test, y_train, y_test;
        test_train::split(m_x, m_y, x_train, x_test, y_train, y_test, split, shuffle);
        lr.train(x_train, y_train);

        std::cout << "  weights: " << lr.weights().transpose() << std::endl;
        std::cout << "  score: " << lr.score(x_test, y_test) << std::endl;
    }

    // regression with polynoms of features
    polynomial_features pf (2, true);
    m_x.resize(data.rows(), 3);
    m_x.col(0) = data.col(4);
    m_x.col(1) = data.col(5);
    m_x.col(2) = data.col(6);
    auto m_x_poly = pf.transform(m_x);

    {
        std::cout << "regression with polynomial features and train-test-split:" << std::endl;

        Eigen::MatrixXd x_train, x_test, y_train, y_test;
        test_train::split(m_x_poly, m_y, x_train, x_test, y_train, y_test, split, shuffle);
        lr.train(x_train, y_train);

        std::cout << "  weights: " << lr.weights().transpose() << std::endl;
        std::cout << "  score: " << lr.score(x_test, y_test) << std::endl;
    }
}
