#include <classification/oneforone.h>
#include <classification/logreg.h>
#include <classification/multinomial_logreg.h>
#include <core/traintest.h>
#include <core/normalize.h>
#include <core/csv_reader.h>
#include <iostream>
#include <numeric>
#include <chrono>

int main(int argc, char** args)
{
    Eigen::MatrixXd x_datas;
    std::vector<std::string> m_classes;
    Eigen::MatrixXd y_datas;

    csv_reader csv(
        [&](size_t lines) {
            x_datas = Eigen::MatrixXd::Zero(lines, 5);
            y_datas = Eigen::MatrixXd::Zero(lines, 1);
        }, 
        [&](size_t line, std::vector<std::string> tokens) {
            for (auto i = 0; i < 5; i++)
                x_datas(line, i) = stod(tokens[i + 3]);
            auto class_name = tokens[8];
            auto it = std::find(m_classes.begin(), m_classes.end(), class_name);
            if (it == m_classes.end()) {
                y_datas(line, 0) = m_classes.size();
                m_classes.push_back(class_name);
            } else
                y_datas(line, 0) = it - m_classes.begin();
        });
    std::cout << "reading data ... ";
    csv.read("foods.csv");
    std::cout << "done" << std::endl;

    x_datas = normalize::transform(x_datas);

    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    test_train::split(x_datas, y_datas, x_train, x_test, y_train, y_test);
    
    logistic_regression lr;
    auto s_ova = std::chrono::high_resolution_clock::now();
    lr.train(x_train, y_train, m_classes);
    auto e_ova = std::chrono::high_resolution_clock::now();
    auto dur_ova = std::chrono::duration_cast<std::chrono::milliseconds> (e_ova - s_ova);
    std::cout << "score - one vs all: " << lr.score(x_test, y_test) << ", training took " << dur_ova.count() << "ms" << std::endl;

    one_for_one ofo;
    auto s_ovo = std::chrono::high_resolution_clock::now();
    ofo.train(x_train, y_train, m_classes);
    auto e_ovo = std::chrono::high_resolution_clock::now();
    auto dur_ovo = std::chrono::duration_cast<std::chrono::milliseconds> (e_ovo - s_ovo);
    std::cout << "score - one vs one: " << ofo.score(x_test, y_test) << ", training took " << dur_ovo.count() << "ms" << std::endl;

    multinomial_logistic_regression mlr;
    auto s_mlr = std::chrono::high_resolution_clock::now();
    mlr.train(x_train, y_train, m_classes);
    auto e_mlr = std::chrono::high_resolution_clock::now();
    auto dur_mlr = std::chrono::duration_cast<std::chrono::milliseconds> (e_mlr - s_mlr);
    std::cout << "score - multinomial: " << mlr.score(x_test, y_test) << ", training took " << dur_mlr.count() << "ms" << std::endl;

    return 0;
}
