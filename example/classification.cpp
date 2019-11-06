#include <classification/oneforone.h>
#include <classification/logreg.h>
#include <classification/multinomial_logreg.h>
#include <classification/knn.h>
#include <classification/decision_tree.h>
#include <classification/random_forest.h>
#include <classification/naive_bayes_gauss.h>
#include <core/traintest.h>
#include <core/normalize.h>
#include <core/csv_data.h>
#include <iostream>
#include <numeric>
#include <chrono>

void do_classification(classifier *c, const std::string& name, size_t classes, const Eigen::MatrixXd xtrain, const Eigen::MatrixXd xtest, const Eigen::MatrixXd ytrain, const Eigen::MatrixXd ytest) 
{
    auto start = std::chrono::high_resolution_clock::now();
    c->init_classes(classes);
    c->train(xtrain, ytrain);
    double score = c->score(xtest, ytest);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    std::cout << "score - " << name << ": " << score << ", took " << dur.count() << "ms" << std::endl;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, size_t> read_foods()
{
    std::cout << "reading data ... ";
    csv_data data;
    data.read("foods.csv");

    Eigen::MatrixXd x_datas = data.matrixFromCols({3, 4, 5, 6, 7});
    Eigen::MatrixXd y_datas = data.matrixFromCols({8}, csv_data::UniqueStringIndex);
    std::cout << "done" << std::endl;
    return std::make_tuple(x_datas, y_datas, static_cast<size_t>(y_datas.maxCoeff() + 1));


    // Eigen::MatrixXd x_datas;
    // std::vector<std::string> classes;
    // Eigen::MatrixXd y_datas;

    // csv_reader csv(
    //     [&](size_t lines) {
    //         x_datas = Eigen::MatrixXd::Zero(lines, 5);
    //         y_datas = Eigen::MatrixXd::Zero(lines, 1);
    //     }, 
    //     [&](size_t line, std::vector<std::string> tokens) {
    //         for (auto i = 0; i < 5; i++)
    //             x_datas(line, i) = stod(tokens[i + 3]);
    //         auto class_name = tokens[8];
    //         auto it = std::find(classes.begin(), classes.end(), class_name);
    //         if (it == classes.end()) {
    //             y_datas(line, 0) = classes.size();
    //             classes.push_back(class_name);
    //         } else
    //             y_datas(line, 0) = it - classes.begin();
    //     });
    
    // csv.read("foods.csv");
    
    // return std::make_tuple(x_datas, y_datas, classes);
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, size_t> read_cancer()
{
    std::cout << "reading data ... ";
    csv_data data;
    data.read("cancer.csv");

    std::vector<size_t> columns(30);
    std::iota(columns.begin(), columns.end(), 2);

    Eigen::MatrixXd x_datas = data.matrixFromCols(columns);
    Eigen::MatrixXd y_datas = data.matrixFromCols({1}, csv_data::UniqueStringIndex);
    std::cout << "done" << std::endl;
    return std::make_tuple(x_datas, y_datas, static_cast<size_t>(y_datas.maxCoeff() + 1));

    // Eigen::MatrixXd x_datas;
    // std::vector<std::string> classes;
    // Eigen::MatrixXd y_datas;

    // csv_reader csv(
    //     [&](size_t lines) {
    //         x_datas = Eigen::MatrixXd::Zero(lines, 30);
    //         y_datas = Eigen::MatrixXd::Zero(lines, 1);
    //     }, 
    //     [&](size_t line, std::vector<std::string> tokens) {
    //         for (auto i = 0; i < 30; i++)
    //             x_datas(line, i) = stod(tokens[i + 2]);
    //         auto class_name = tokens[1];
    //         auto it = std::find(classes.begin(), classes.end(), class_name);
    //         if (it == classes.end()) {
    //             y_datas(line, 0) = classes.size();
    //             classes.push_back(class_name);
    //         } else
    //             y_datas(line, 0) = it - classes.begin();
    //     });
    // std::cout << "reading data ... ";
    // csv.read("cancer.csv");
    // std::cout << "done" << std::endl;
    // return std::make_tuple(x_datas, y_datas, classes);
}

int main(int argc, char** args)
{
    // auto [x_datas, y_datas, classes] = read_foods();
    auto [x_datas, y_datas, classes] = read_cancer();

    x_datas = normalize::transform(x_datas);

    Eigen::MatrixXd x_train, x_test, y_train, y_test;
    do_train_test_split(x_datas, y_datas, x_train, x_test, y_train, y_test, 0.25, true);
    double score;

    logistic_regression lr;
    do_classification(&lr, "logistic regression (one vs all)", classes, x_train, x_test, y_train, y_test);

    one_for_one ofo;
    do_classification(&ofo, "logistic regression (one vs one)", classes, x_train, x_test, y_train, y_test);

    multinomial_logistic_regression mlr;
    do_classification(&mlr, "multinomial", classes, x_train, x_test, y_train, y_test);

    knn k;
    k.set_param("k", 3);
    do_classification(&k, "k nearest neighbours", classes, x_train, x_test, y_train, y_test);

    decision_tree dt;
    dt.set_param("max_depth", 10);
    dt.set_param("min_leaf_items", 1);
    do_classification(&dt, "decision tree", classes, x_train, x_test, y_train, y_test);

    random_forest rf;
    rf.set_param("trees", 5);
    rf.set_param("max_depth", 10);
    rf.set_param("min_leaf_items", 1);
    rf.set_param("ignored_features", 1);
    do_classification(&rf, "random forest", classes, x_train, x_test, y_train, y_test);

    naive_bayes_gauss nbg;
    do_classification(&nbg, "naive bayes (gauss)", classes, x_train, x_test, y_train, y_test);

    return 0;
}
