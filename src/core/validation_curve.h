#ifndef _VALIDATION_CURVE_H_
#define _VALIDATION_CURVE_H_

#include <vector>
#include <string>
#include <Eigen/Dense>

class model;

class validation_curve
{
public:
    validation_curve() = default;
    validation_curve(const validation_curve&) = delete;
    ~validation_curve() = default;

    static Eigen::MatrixXd create(model *m, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::string& parameter, std::vector<double> parameter_values);
};

#endif