#include <vector>

class linear_regression
{
public:
    linear_regression();
    ~linear_regression() = default;

    void train(const std::vector<std::vector<double>> &x, const std::vector<double> &y);

    std::vector<double> coeffs();

    double predict(const std::vector<double> &x);

private:
    double cost(const std::vector<std::vector<double>> &x, const std::vector<double> &y);
    void init_coeffs(size_t count);
    void init_buffers(size_t count);

    double m_rate;
    double m_threshold;

    std::vector<double> m_coeffs;
    std::vector<double> m_coeffs_delta;
    std::vector<double> m_sx;
    std::vector<double> m_sxx;
    std::vector<double> m_sxy;
};