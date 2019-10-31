#include <regression/linreg.h>

#include <gtest/gtest.h>
#include <numeric>
#include <random>

static const size_t test_size = 100;

class linear_regression_test : public ::testing::Test {
public:
    linear_regression_test() = default;
    ~linear_regression_test() = default;

protected:
    std::vector<double> generate_data(size_t samples, size_t iota_start, double factor, bool add_noise, double noise_sd)
    {
        std::vector<double> data (samples);

        std::iota(data.begin(), data.end(), iota_start);
        std::transform(data.begin(), data.end(), data.begin(), std::bind(std::multiplies<>(), std::placeholders::_1, factor));

        if (add_noise) {
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<> d{0, noise_sd};
            std::transform(data.begin(), data.end(), data.begin(), [&](const double &a) { return a + d(gen); });
        }

        return data;
    }

    linear_regression m_linreg;
};

TEST_F (linear_regression_test, constructor_coeffs_uninitialized) { 
    ASSERT_EQ(Eigen::MatrixXd() , m_linreg.weights());
}

TEST_F (linear_regression_test, predicts_correct_line) { 
    auto x = generate_data(test_size, 0, 1.0, true, 0.0005);
    auto y = generate_data(test_size, 10, 1.0, true, 0.0005);

    Eigen::MatrixXd m_y(y.size(), 1);
    Eigen::MatrixXd m_x(x.size(), 2);
    m_y.col(0) = Eigen::VectorXd::Map(y.data(), y.size());
    m_x.col(0) = Eigen::VectorXd::Map(x.data(), x.size());
    m_x.col(1) = Eigen::VectorXd::Ones(x.size());

    m_linreg.train(m_x, m_y);
    ASSERT_EQ(2, m_linreg.weights().rows());
    ASSERT_NEAR(1, m_linreg.weights()(0, 0), 0.001);
    ASSERT_NEAR(10, m_linreg.weights()(1, 0), 0.001);

    Eigen::MatrixXd x_test(1, 2);
    x_test << 100.0, 1.0;
    ASSERT_NEAR(110, m_linreg.predict(x_test)(0,0), 0.001);
}