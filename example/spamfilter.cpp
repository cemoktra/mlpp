#include <classification/naive_bayes.h>
#include <classification/gauss_distribution.h>
#include <classification/binomial_distribution.h>
#include <core/traintest.h>
#include <core/csv_data.h>
#include <core/vocabulary.h>
#include <iostream>
#include <numeric>
#include <chrono>

int main(int argc, char** args)
{
    csv_data csv;
    std::cout << "reading data ... ";
    csv.read("spam.csv");
    std::cout << "done" << std::endl;

    train_test_split tts;
    std::vector<std::string> test_strings, train_strings;
    tts.init(csv.rows());
    tts.split(csv.col<std::string>(1), train_strings, test_strings);

    std::cout << "preparing vocabulary ... ";
    vocabulary voc;
    voc.add(train_strings);
    voc.filter_counts(3, 50);
    std::cout << "done" << std::endl;

    std::cout << "preparing data ... ";
    Eigen::MatrixXd y_train, y_test;
    auto y = csv.matrixFromCols({ 0 }, csv_data::UniqueStringIndex);
    auto x_train = voc.transform(train_strings);
    auto x_test = voc.transform(test_strings);
    tts.split(y, y_train, y_test);
    std::cout << "done" << std::endl;

    double score;

    naive_bayes nbb (std::make_shared<binomial_distribution>());
    auto s_nbb = std::chrono::high_resolution_clock::now();
    nbb.init_classes(static_cast<size_t>(y.maxCoeff() + 1));
    nbb.train(x_train, y_train);
    score = nbb.score(x_test, y_test);
    auto e_nbb = std::chrono::high_resolution_clock::now();
    auto dur_nbb = std::chrono::duration_cast<std::chrono::milliseconds> (e_nbb - s_nbb);
    std::cout << "naive bayes binomial score: " << score << ", took " << dur_nbb.count() << "ms" << std::endl;


    for (auto k = 0; k < x_train.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(x_train, k); it; ++it)
            it.valueRef() = 1.0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(x_test, k); it; ++it)
            it.valueRef() = 1.0;
    }
    
    naive_bayes nbg (std::make_shared<gauss_distribution>());
    auto s_nbg = std::chrono::high_resolution_clock::now();
    nbg.init_classes(static_cast<size_t>(y.maxCoeff() + 1));
    nbg.train(x_train, y_train);
    score = nbg.score(x_test, y_test);
    auto e_nbg = std::chrono::high_resolution_clock::now();
    auto dur_nbg = std::chrono::duration_cast<std::chrono::milliseconds> (e_nbg - s_nbg);
    std::cout << "naive bayes gauss score: " << score << ", took " << dur_nbg.count() << "ms" << std::endl;

    return 0;
}
