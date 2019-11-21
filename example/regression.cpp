#include <regression/linreg.h>
#include <core/traintest.h>
#include <core/kfold.h>
#include <core/csv_data.h>
#include <core/polyfeatures.h>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <iostream>

std::pair<xt::xarray<double>, xt::xarray<double>> read_diamonds()
{
    std::cout << "reading data ... ";
    csv_data data;
    data.read("diamonds.csv");

    xt::xarray<double> y = data.matrixFromCols({6});
    xt::xarray<double> X = data.matrixFromCols({0, 7, 8, 9});

    std::cout << "done" << std::endl;
    return std::pair(X, y);
}

int main(int argc, char** args)
{
    auto [x_datas, y_datas] = read_diamonds();

    linear_regression lr;
    bool shuffle = true;
    double split = 0.25;
    
    xt::xarray<double> X_subset = xt::ones<double>(std::vector<size_t>({ x_datas.shape()[0], 2 }));
    xt::view(X_subset, xt::all(), xt::range(0, 1)) = xt::view(x_datas, xt::all(), xt::range(0, 1));
    xt::xarray<double> X_train, X_test, y_train, y_test;

    // regression train-test-split
    {
        std::cout << "regression with one feature and train-test-split:" << std::endl;
        
        do_train_test_split(X_subset, y_datas, X_train, X_test, y_train, y_test, split, shuffle);
        lr.train(X_train, y_train);
        std::cout << "  weights: " << xt::transpose(lr.weights()) << std::endl;
        std::cout << "  score: " << lr.score(X_test, y_test) << std::endl;
    }

    // regression kfold
    {
        std::cout << "regression with one feature kfold:" << std::endl;
        kfold kf(4, shuffle);

        xt::xarray<double> mean_weights;

        for (auto i = 0; i < kf.k(); ++i) {
            kf.split(i, X_subset, y_datas, X_train, X_test, y_train, y_test);
            lr.train(X_train, y_train);
            
            std::cout << "  fold " << i + 1 << std::endl;
            std::cout << "    coeffs: " << xt::transpose(lr.weights()) << std::endl;
            std::cout << "    score: " << lr.score(X_test, y_test) << std::endl;

            if (i > 0)
                mean_weights += lr.weights();
            else 
                mean_weights = lr.weights();
        }
        mean_weights /= kf.k();

        lr.set_weights(mean_weights);
        std::cout << "  mean coeffs: ";
        std::cout << xt::transpose(lr.weights()) << std::endl;
        std::cout << "  mean score: " << lr.score(X_test, y_test) << std::endl;
    }

    // regression with multi vars
    X_subset = xt::ones<double>(std::vector<size_t>({ x_datas.shape()[0], 4 }));
    xt::view(X_subset, xt::all(), xt::range(0, 3)) = xt::view(x_datas, xt::all(), xt::range(1, 4));
    
    {
        std::cout << "regression with multiple features and train-test-split:" << std::endl;

        do_train_test_split(X_subset, y_datas, X_train, X_test, y_train, y_test, split, shuffle);
        lr.train(X_train, y_train);

        std::cout << "  weights: " << xt::transpose(lr.weights()) << std::endl;
        std::cout << "  score: " << lr.score(X_test, y_test) << std::endl;
    }

    // regression with polynoms of features
    polynomial_features pf (2, true);
    X_subset = xt::ones<double>(std::vector<size_t>({ x_datas.shape()[0], 3 }));
    xt::view(X_subset, xt::all(), xt::range(0, 3)) = xt::view(x_datas, xt::all(), xt::range(1, 4));
    xt::xarray<double> X_poly = pf.transform(X_subset);

    {
        std::cout << "regression with polynomial features and train-test-split:" << std::endl;

        do_train_test_split(X_poly, y_datas, X_train, X_test, y_train, y_test, split, shuffle);
        lr.train(X_train, y_train);

        std::cout << "  weights: " << xt::transpose(lr.weights()) << std::endl;
        std::cout << "  score: " << lr.score(X_test, y_test) << std::endl;
    }
}
