#include <classification/oneforone.h>
#include <classification/logreg.h>
#include <classification/multinomial_logreg.h>
#include <classification/knn.h>
#include <classification/decision_tree.h>
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
    double score;

    logistic_regression lr;
    auto s_ova = std::chrono::high_resolution_clock::now();
    lr.train(x_train, y_train, m_classes);
    score = lr.score(x_test, y_test);
    auto e_ova = std::chrono::high_resolution_clock::now();
    auto dur_ova = std::chrono::duration_cast<std::chrono::milliseconds> (e_ova - s_ova);
    std::cout << "score - one vs all: " << score << ", took " << dur_ova.count() << "ms" << std::endl;

    one_for_one ofo;
    auto s_ovo = std::chrono::high_resolution_clock::now();
    ofo.train(x_train, y_train, m_classes);
    score = ofo.score(x_test, y_test);
    auto e_ovo = std::chrono::high_resolution_clock::now();
    auto dur_ovo = std::chrono::duration_cast<std::chrono::milliseconds> (e_ovo - s_ovo);
    std::cout << "score - one vs one: " << score << ", took " << dur_ovo.count() << "ms" << std::endl;

    multinomial_logistic_regression mlr;
    auto s_mlr = std::chrono::high_resolution_clock::now();
    mlr.train(x_train, y_train, m_classes);
    score = mlr.score(x_test, y_test);
    auto e_mlr = std::chrono::high_resolution_clock::now();
    auto dur_mlr = std::chrono::duration_cast<std::chrono::milliseconds> (e_mlr - s_mlr);
    std::cout << "score - multinomial: " << score << ", took " << dur_mlr.count() << "ms" << std::endl;

    knn k (3);
    auto s_knn = std::chrono::high_resolution_clock::now();
    k.train(x_train, y_train, m_classes);
    score = k.score(x_test, y_test);
    auto e_knn = std::chrono::high_resolution_clock::now();
    auto dur_knn = std::chrono::duration_cast<std::chrono::milliseconds> (e_knn - s_knn);
    std::cout << "score - knn: " << score << ", took " << dur_mlr.count() << "ms" << std::endl;

    decision_tree dt;
    auto s_dt = std::chrono::high_resolution_clock::now();
    dt.train(x_train, y_train, m_classes);
    score = dt.score(x_test, y_test);
    auto e_dt = std::chrono::high_resolution_clock::now();
    auto dur_dt = std::chrono::duration_cast<std::chrono::milliseconds> (e_dt - s_dt);
    std::cout << "score - dt: " << score << ", took " << dur_dt.count() << "ms" << std::endl;

    return 0;
}
