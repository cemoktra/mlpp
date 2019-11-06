#include "pseudo_inverse.h"

Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& mat, double epsilon)
{
    auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &singularValues = svd.singularValues();
    Eigen::MatrixXd singularValuesInv(mat.cols(), mat.rows());
    singularValuesInv.setZero();
    for (unsigned int i = 0; i < singularValues.size(); ++i)
        singularValuesInv(i, i) = singularValues(i) > epsilon ? 1.0 / singularValues(i) : 0.0;
    return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
}