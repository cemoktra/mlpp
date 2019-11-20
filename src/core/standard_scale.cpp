
#include "standard_scale.h"

Eigen::MatrixXd standard_scale::transform(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd result = x;
    for (auto i = 0; i < result.cols(); ++i)
    {
        auto col_array = result.col(i).array();
        auto mean = col_array.mean();
        auto std_dev = std::sqrt((col_array - mean).square().sum() / (col_array.size() - 1));
        col_array -= mean;
        col_array /= std_dev;
    }
    return result;
}