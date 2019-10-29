
#include "normalize.h"
#include <math.h>
#include <cstdlib>

Eigen::MatrixXd normalize::transform(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd result = x;

    for (auto i = 0; i < result.cols(); ++i)
    {
        auto mean = result.col(i).mean();
        result.col(i) -= Eigen::VectorXd::Constant(result.rows(), mean);
        auto min = result.col(i).minCoeff();
        auto max = result.col(i).maxCoeff();
        auto scale = std::max(std::fabs(min), std::fabs(max));
        result.col(i) /= scale;
    }

    return result;
}