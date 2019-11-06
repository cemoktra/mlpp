
#include "one_hot.h"

Eigen::MatrixXd one_hot::transform(const Eigen::MatrixXd& x, size_t classes)
{
    Eigen::MatrixXd result (x.rows(), classes);
    for (auto i = 0; i < classes; i++)
        result.col(i) = (x.array() == i).cast<double>();
    return result;
}