
#include "one_hot.h"

Eigen::MatrixXd one_hot::transform(const Eigen::MatrixXd& x)
{
    std::vector<double> xvec (x.data(), x.data() + x.rows() * x.cols());
    std::sort(xvec.begin(), xvec.end());
    size_t classes = std::unique(xvec.begin(), xvec.end()) - xvec.begin();

    Eigen::MatrixXd result (x.rows(), classes);
    for (auto i = 0; i < classes; i++)
        result.col(i) = (x.array() == xvec[i]).cast<double>();
    
    return result;
}