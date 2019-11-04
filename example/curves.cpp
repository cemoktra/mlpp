#include <classification/logreg.h>
#include <core/normalize.h>
#include <core/csv_reader.h>
#include <core/validation_curve.h>
#include <iostream>
#include <numeric>
#include <chrono>

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
    auto [x_datas, y_datas, classes] = read_cancer();

    x_datas = normalize::transform(x_datas);

    logistic_regression m;
    m.init_classes(classes);
    Eigen::MatrixXd result = validation_curve::create(&m, x_datas, y_datas, "max_iterations", { 1, 10, 100, 1000, 10000});

    std::cout << result.transpose() << std::endl;

    return 0;
}
