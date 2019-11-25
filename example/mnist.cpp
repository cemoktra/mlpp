
#include <classification/naive_bayes.h>
#include <classification/gauss_distribution.h>
#include <core/mnist_data.h>
#include <core/normalize.h>
#include <iostream>
#include <chrono>

int main(int argc, char** args)
{
    double score;

    std::cout << "reading data ... ";
    mnist_data data;
    auto [X_train, X_test, y_train, y_test] = data.read("./mnist/", "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");
    std::cout << "done" << std::endl;

    // make images one dimensional
    X_train.reshape(std::vector<size_t>({X_train.shape()[0], X_train.shape()[1] * X_train.shape()[1]}));
    X_test.reshape(std::vector<size_t>({X_test.shape()[0], X_test.shape()[1] * X_test.shape()[1]}));
    // normalize data
    X_train = normalize::transform(X_train);
    X_test  = normalize::transform(X_test);

    naive_bayes nbg (std::make_shared<gauss_distribution>());
    auto s_nbg = std::chrono::high_resolution_clock::now();
    nbg.init_classes(static_cast<size_t>(xt::amax(y_train)(0) + 1));
    nbg.train(X_train, y_train);
    score = nbg.score(X_test, y_test);
    auto e_nbg = std::chrono::high_resolution_clock::now();
    auto dur_nbg = std::chrono::duration_cast<std::chrono::milliseconds> (e_nbg - s_nbg);
    std::cout << "naive bayes gauss score: " << score << ", took " << dur_nbg.count() << "ms" << std::endl;

    return 0;
}