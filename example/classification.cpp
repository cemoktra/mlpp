#include <classification/oneforone.h>
#include <classification/logreg.h>
#include <classification/multinomial_logreg.h>
#include <classification/knn.h>
#include <classification/decision_tree.h>
#include <classification/random_forest.h>
#include <core/traintest.h>
#include <core/normalize.h>
#include <core/csv_reader.h>
#include <iostream>
#include <numeric>
#include <chrono>

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::string>> read_foods()
{
    Eigen::MatrixXd x_datas;
    std::vector<std::string> classes;
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
            auto it = std::find(classes.begin(), classes.end(), class_name);
            if (it == classes.end()) {
                y_datas(line, 0) = classes.size();
                classes.push_back(class_name);
            } else
                y_datas(line, 0) = it - classes.begin();
        });
    std::cout << "reading data ... ";
    csv.read("foods.csv");
    std::cout << "done" << std::endl;
    return std::make_tuple(x_datas, y_datas, classes);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<std::string>> read_cancer()
{
    Eigen::MatrixXd x_datas;
    std::vector<std::string> classes;
    Eigen::MatrixXd y_datas;

    csv_reader csv(
        [&](size_t lines) {
            x_datas = Eigen::MatrixXd::Zero(lines, 30);
            y_datas = Eigen::MatrixXd::Zero(lines, 1);
        }, 
        [&](size_t line, std::vector<std::string> tokens) {
            for (auto i = 0; i < 30; i++)
                x_datas(line, i) = stod(tokens[i + 2]);
            auto class_name = tokens[1];
            auto it = std::find(classes.begin(), classes.end(), class_name);
            if (it == classes.end()) {
                y_datas(line, 0) = classes.size();
                classes.push_back(class_name);
            } else
                y_datas(line, 0) = it - classes.begin();
        });
    std::cout << "reading data ... ";
    csv.read("cancer.csv");
    std::cout << "done" << std::endl;
    return std::make_tuple(x_datas, y_datas, classes);
}

int main(int argc, char** args)
{
    auto [x_datas, y_datas, classes] = read_foods();
    // auto [x_datas, y_datas, classes] = read_cancer();

    x_datas = normalize::transform(x_datas);

    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    test_train::split(x_datas, y_datas, x_train, x_test, y_train, y_test);
    double score;

    logistic_regression lr;
    auto s_ova = std::chrono::high_resolution_clock::now();
    lr.init_classes(classes);
    lr.train(x_train, y_train);
    score = lr.score(x_test, y_test);
    auto e_ova = std::chrono::high_resolution_clock::now();
    auto dur_ova = std::chrono::duration_cast<std::chrono::milliseconds> (e_ova - s_ova);
    std::cout << "score - logistic regression (one vs all): " << score << ", took " << dur_ova.count() << "ms" << std::endl;

    one_for_one ofo;
    auto s_ovo = std::chrono::high_resolution_clock::now();
    ofo.init_classes(classes);
    ofo.train(x_train, y_train);
    score = ofo.score(x_test, y_test);
    auto e_ovo = std::chrono::high_resolution_clock::now();
    auto dur_ovo = std::chrono::duration_cast<std::chrono::milliseconds> (e_ovo - s_ovo);
    std::cout << "score - logistic regression (one vs one): " << score << ", took " << dur_ovo.count() << "ms" << std::endl;

    multinomial_logistic_regression mlr;
    auto s_mlr = std::chrono::high_resolution_clock::now();
    mlr.init_classes(classes);
    mlr.train(x_train, y_train);
    score = mlr.score(x_test, y_test);
    auto e_mlr = std::chrono::high_resolution_clock::now();
    auto dur_mlr = std::chrono::duration_cast<std::chrono::milliseconds> (e_mlr - s_mlr);
    std::cout << "score - multinomial: " << score << ", took " << dur_mlr.count() << "ms" << std::endl;

    knn k;
    k.set_param("k", 3);
    auto s_knn = std::chrono::high_resolution_clock::now();
    k.init_classes(classes);
    k.train(x_train, y_train);
    score = k.score(x_test, y_test);
    auto e_knn = std::chrono::high_resolution_clock::now();
    auto dur_knn = std::chrono::duration_cast<std::chrono::milliseconds> (e_knn - s_knn);
    std::cout << "score - k nearest neighbours: " << score << ", took " << dur_mlr.count() << "ms" << std::endl;

    decision_tree dt;
    dt.set_param("max_depth", 10);
    dt.set_param("min_leaf_items", 1);
    auto s_dt = std::chrono::high_resolution_clock::now();
    dt.init_classes(classes);
    dt.train(x_train, y_train);
    score = dt.score(x_test, y_test);
    auto e_dt = std::chrono::high_resolution_clock::now();
    auto dur_dt = std::chrono::duration_cast<std::chrono::milliseconds> (e_dt - s_dt);
    std::cout << "score - decision tree: " << score << ", took " << dur_dt.count() << "ms" << std::endl;

    random_forest rf;
    rf.set_param("trees", 5);
    rf.set_param("max_depth", 10);
    rf.set_param("min_leaf_items", 1);
    rf.set_param("ignored_features", 1);    
    auto s_rf = std::chrono::high_resolution_clock::now();
    rf.init_classes(classes);
    rf.train(x_train, y_train);
    score = rf.score(x_test, y_test);
    auto e_rf = std::chrono::high_resolution_clock::now();
    auto dur_rf = std::chrono::duration_cast<std::chrono::milliseconds> (e_rf - s_rf);
    std::cout << "score - random forest: " << score << ", took " << dur_rf.count() << "ms" << std::endl;

    return 0;
}
