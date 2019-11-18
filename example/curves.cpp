#include <classification/logreg.h>
#include <core/normalize.h>
#include <core/csv_data.h>
#include <core/validation_curve.h>
#include <core/learning_curve.h>
#include <iostream>
#include <numeric>
#include <chrono>

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
}

int main(int argc, char** args)
{
    auto [x_datas, y_datas, classes] = read_cancer();

    x_datas = normalize::transform(x_datas);

    logistic_regression m;
    m.init_classes(classes);

    Eigen::MatrixXd result = learning_curve::create(&m, x_datas, y_datas);
    std::cout << result.transpose() << std::endl;

    result = validation_curve::create(&m, x_datas, y_datas, "max_iterations", { 1, 10, 100, 1000, 10000});
    std::cout << result.transpose() << std::endl;

    return 0;
}
