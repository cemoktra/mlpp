
#include "one_hot.h"

Eigen::MatrixXd one_hot::transform(const Eigen::MatrixXd& x, const std::vector<std::string>& classes)
{
    size_t class_count = classes.size();

    Eigen::MatrixXd result (x.rows(), class_count);
    for (auto i = 0; i < class_count; i++)
        result.col(i) = (x.array() == i).cast<double>();
    return result;
}