
#include <classification/naive_bayes.h>
#include <classification/gauss_distribution.h>
#include <core/mnist_data.h>
#include <core/standard_scale.h>
#include <core/scoped_timer.h>
#include <iostream>
#include <chrono>
#include <xtensor/xio.hpp>

int main(int argc, char** args)
{
    double score;

    scoped_timer *st = new scoped_timer("reading data");
    mnist_data data;
    auto [X_train, X_test, y_train, y_test] = data.read("./mnist/", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");
    delete st;

    // make images one dimensional
    X_train.reshape(std::vector<size_t>({X_train.shape()[0], X_train.shape()[1] * X_train.shape()[1]}));
    X_test.reshape(std::vector<size_t>({X_test.shape()[0], X_test.shape()[1] * X_test.shape()[1]}));
    // standard_scale data
    st = new scoped_timer("standard_scale");
    X_train = standard_scale::transform(X_train);
    X_test  = standard_scale::transform(X_test);
    delete st;

    naive_bayes nbg (std::make_shared<gauss_distribution>());
    nbg.init_classes(static_cast<size_t>(xt::amax(y_train)(0) + 1));
    st = new scoped_timer("naive bayes (gaussian) - train");
    nbg.train(X_train, y_train);
    delete st;
    st = new scoped_timer("naive bayes (gaussian) - score");
    score = nbg.score(X_test, y_test);
    delete st;
    std::cout << "=> naive bayes (gaussian) score: " << score << std::endl;

    return 0;
}