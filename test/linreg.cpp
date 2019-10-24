#include "../src/linreg.h"

#include <gtest/gtest.h>
#include <numeric>

class linear_regression_test : public ::testing::Test {
public:
    linear_regression_test() = default;
    ~linear_regression_test() = default;

protected:
    linear_regression m_linreg;
};

TEST_F (linear_regression_test, constructor_coeffs_uninitialized) { 
    ASSERT_EQ(0, m_linreg.coeffs().size());
}

TEST_F (linear_regression_test, predicts_correct_line) { 
    std::vector<double> x0, y;
    std::vector<std::vector<double>> x;

    x0.resize(50);
    y.resize(50);
    std::iota(x0.begin(), x0.end(), 0);
    std::iota(y.begin(), y.end(), 10);
    x.push_back(x0);

    m_linreg.train(x, y);

    ASSERT_EQ(2, m_linreg.coeffs().size());
    
    ASSERT_EQ(2, m_linreg.coeffs().size());
    ASSERT_NEAR(1, m_linreg.coeffs()[0], 0.001);
    ASSERT_NEAR(10, m_linreg.coeffs()[1], 0.001);

    std::vector<double> x_test;
    x_test.push_back(100.0);
    ASSERT_NEAR(110, m_linreg.predict(x_test), 0.001);
}