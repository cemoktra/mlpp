#ifndef _PSEUDO_INVERSE_H_
#define _PSEUDO_INVERSE_H_

#include <Eigen/Dense>

Eigen::MatrixXd pseudo_inverse(const Eigen::MatrixXd& mat, double epsilon = 1e-6);

#endif