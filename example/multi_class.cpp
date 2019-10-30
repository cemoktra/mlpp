#include "../src/oneforone.h"
#include "../src/logreg.h"
#include "../src/multinomial_logreg.h"
#include "../src/traintest.h"
#include "../src/normalize.h"
#include "../src/one_hot.h"
#include "../test/testdata.h"
#include <iostream>
#include <numeric>
#include <chrono>

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
    test_train::split(m_x, m_y, x_train, x_test, y_train, y_test, 0.25, true);
    
    logistic_regression lr;
    auto s_ova = std::chrono::high_resolution_clock::now();
    lr.train(x_train, y_train);
    auto e_ova = std::chrono::high_resolution_clock::now();
    auto dur_ova = std::chrono::duration_cast<std::chrono::milliseconds> (e_ova - s_ova);
    std::cout << "score - one vs all: " << lr.score(x_test, y_test) << ", training took " << dur_ova.count() << "ms" << std::endl;

    one_for_one ofo;
    auto s_ovo = std::chrono::high_resolution_clock::now();
    ofo.train(x_train, y_train);
    auto e_ovo = std::chrono::high_resolution_clock::now();
    auto dur_ovo = std::chrono::duration_cast<std::chrono::milliseconds> (e_ovo - s_ovo);
    std::cout << "score - one vs one: " << ofo.score(x_test, y_test) << ", training took " << dur_ovo.count() << "ms" << std::endl;

    multinomial_logistic_regression mlr;
    auto s_mlr = std::chrono::high_resolution_clock::now();
    mlr.train(x_train, y_train);
    auto e_mlr = std::chrono::high_resolution_clock::now();
    auto dur_mlr = std::chrono::duration_cast<std::chrono::milliseconds> (e_mlr - s_mlr);
    std::cout << "score - multinomial: " << mlr.score(x_test, y_test) << ", training took " << dur_mlr.count() << "ms" << std::endl;

    return 0;
}
