#ifndef _TRAINTEST_H_
#define _TRAINTEST_H_

#include <vector>

class test_train
{
public:
    test_train() = delete;
    ~test_train() = delete;

    static void split(const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::vector<std::vector<double>>& x_train, std::vector<std::vector<double>>& x_test, std::vector<double>& y_train, std::vector<double>& y_test, double test_proportion = 0.25);
};

#endif