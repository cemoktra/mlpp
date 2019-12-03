#include <classification/logreg.h>
#include <classification/oneforone.h>
#include <classification/multinomial_logreg.h>
#include <classification/knn.h>
#include <classification/decision_tree.h>
#include <classification/random_forest.h>
#include <classification/naive_bayes.h>
#include <classification/gauss_distribution.h>
#include <classification/binomial_distribution.h>
#include <core/traintest.h>
#include <core/standard_scale.h>
#include <core/normalize.h>
#include <core/csv_data.h>
#include <core/scoped_timer.h>
#include <iostream>
#include <numeric>
#include <xtensor/xio.hpp>

void do_classification(classifier *c, const std::string& name, size_t classes, const xt::xarray<double> xtrain, const xt::xarray<double> xtest, const xt::xarray<double> ytrain, const xt::xarray<double> ytest) 
{
    double score;
    xt::xarray<double> confusion;
    {
        scoped_timer st (name);
        auto start = std::chrono::high_resolution_clock::now();
        c->init_classes(classes);
        c->train(xtrain, ytrain);
        confusion = c->confusion(xtest, ytest);
        score = c->score(confusion);
        
    }
    std::cout << "  score = " << score << std::endl;
    // std::cout << "  confusion matrix = " << std::endl;
    // std::cout << confusion << std::endl;
}

std::tuple<xt::xarray<double>, xt::xarray<double>, size_t> read_foods()
{
    scoped_timer st ("reading data");
    csv_data data;
    data.read("foods.csv");

    xt::xarray<double> X = data.matrixFromCols({3, 4, 5, 6, 7});
    xt::xarray<double> y = data.matrixFromCols({8}, csv_data::UniqueStringIndex);
    return std::make_tuple(X, y, static_cast<size_t>(xt::amax(y)(0) + 1));
}

std::tuple<xt::xarray<double>, xt::xarray<double>, size_t> read_cancer()
{
    scoped_timer st ("reading data");
    csv_data data;
    data.read("cancer.csv");

    std::vector<size_t> columns(30);
    std::iota(columns.begin(), columns.end(), 2);

    xt::xarray<double> X = data.matrixFromCols(columns);
    xt::xarray<double> y = data.matrixFromCols({1}, csv_data::UniqueStringIndex);
    return std::make_tuple(X, y, static_cast<size_t>(xt::amax(y)(0) + 1));
}

int main(int argc, char** args)
{
    // auto [X, y, classes] = read_foods();
    auto [X, y, classes] = read_cancer();

    scoped_timer *st = new scoped_timer("preprocessing data");
    auto X_stdscale = standard_scale::transform(X);
    auto X_norm = normalize::transform(X);

    xt::xarray<double> X_train, X_test, y_train, y_test;
    train_test_split tts;
    tts.init(X.shape()[0], 0.25, true);
    std::tie(X_train, X_test, y_train, y_test) = tts.split(X_stdscale, y);
    delete st;

    logistic_regression lr;
    do_classification(&lr, "logistic regression (one vs all)", classes, X_train, X_test, y_train, y_test);

    one_for_one ofo;
    do_classification(&ofo, "logistic regression (one vs one)", classes, X_train, X_test, y_train, y_test);

    multinomial_logistic_regression mlr;
    do_classification(&mlr, "multinomial", classes, X_train, X_test, y_train, y_test);

    knn k;
    k.set_param("k", 3);
    do_classification(&k, "k nearest neighbours", classes, X_train, X_test, y_train, y_test);

    naive_bayes nbg (std::make_shared<gauss_distribution>());
    do_classification(&nbg, "naive bayes (gauss)", classes, X_train, X_test, y_train, y_test);

    // we need normalized data for decision trees
    std::tie(X_train, X_test, y_train, y_test) = tts.split(X_norm, y);

    decision_tree dt;
    dt.set_param("max_depth", 10);
    dt.set_param("min_leaf_items", 1);
    do_classification(&dt, "decision tree", classes, X_train, X_test, y_train, y_test);

    random_forest rf;
    rf.set_param("trees", 5);
    rf.set_param("max_depth", 10);
    rf.set_param("min_leaf_items", 1);
    rf.set_param("ignored_features", 3);
    do_classification(&rf, "random forest", classes, X_train, X_test, y_train, y_test);

    return 0;
}
