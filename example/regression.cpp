#include <regression/linreg.h>
#include <core/traintest.h>
#include <core/kfold.h>
#include <core/csv_reader.h>
#include <core/polyfeatures.h>
#include <iostream>

int main(int argc, char** args)
{
    Eigen::MatrixXd x_datas;
    Eigen::MatrixXd y_datas;

    csv_reader csv(
        [&](size_t lines) {
            x_datas = Eigen::MatrixXd::Ones(lines, 4);
            y_datas = Eigen::MatrixXd::Zero(lines, 1);
        }, 
        [&](size_t line, std::vector<std::string> tokens) {
            try {
            x_datas(line, 0) = stod(tokens[0]);
            for (auto i = 0; i < 3; i++)
                x_datas(line, i + 1) = stod(tokens[i + 7]);
            y_datas(line, 0) = stod(tokens[6]);
            } catch (...) {
                std::cout << "PANIC " << std::endl;
            }
        });
    std::cout << "reading data ... ";
    csv.read("diamonds.csv");
    std::cout << "done" << std::endl;

    linear_regression lr;
    bool shuffle = true;
    double split = 0.25;
    
    Eigen::MatrixXd x_datas_subset (x_datas.rows(), 2);
    x_datas_subset.col(0) = x_datas.col(0);
    x_datas_subset.col(1) = Eigen::MatrixXd::Ones(x_datas.rows(), 1);

    // regression train-test-split
    {
        std::cout << "regression with one feature and train-test-split:" << std::endl;

        Eigen::MatrixXd x_train, x_test, y_train, y_test;
        do_train_test_split(x_datas_subset, y_datas, x_train, x_test, y_train, y_test, split, shuffle);
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
            kf.split(i, x_datas_subset, y_datas, x_train, x_test, y_train, y_test);
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
    x_datas_subset.resize (x_datas.rows(), 4);
    x_datas_subset.col(0) = x_datas.col(1);
    x_datas_subset.col(1) = x_datas.col(2);
    x_datas_subset.col(2) = x_datas.col(3);
    x_datas_subset.col(3) = Eigen::MatrixXd::Ones(x_datas.rows(), 1);
    
    {
        std::cout << "regression with multiple features and train-test-split:" << std::endl;

        Eigen::MatrixXd x_train, x_test, y_train, y_test;
        do_train_test_split(x_datas_subset, y_datas, x_train, x_test, y_train, y_test, split, shuffle);
        lr.train(x_train, y_train);

        std::cout << "  weights: " << lr.weights().transpose() << std::endl;
        std::cout << "  score: " << lr.score(x_test, y_test) << std::endl;
    }

    // regression with polynoms of features
    polynomial_features pf (2, true);
    x_datas_subset.resize (x_datas.rows(), 3);
    x_datas_subset.col(0) = x_datas.col(1);
    x_datas_subset.col(1) = x_datas.col(2);
    x_datas_subset.col(2) = x_datas.col(3);
    auto m_x_poly = pf.transform(x_datas_subset);

    {
        std::cout << "regression with polynomial features and train-test-split:" << std::endl;

        Eigen::MatrixXd x_train, x_test, y_train, y_test;
        do_train_test_split(m_x_poly, y_datas, x_train, x_test, y_train, y_test, split, shuffle);
        lr.train(x_train, y_train);

        std::cout << "  weights: " << lr.weights().transpose() << std::endl;
        std::cout << "  score: " << lr.score(x_test, y_test) << std::endl;
    }
}
