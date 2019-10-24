#ifndef _LINREG_H_
#define _LINREG_H_

#include <vector>

class linear_regression
{
public:
    linear_regression();
    ~linear_regression() = default;

    void train(const std::vector<std::vector<double>> &x, const std::vector<double> &y, size_t maxIterations = 0);

    std::vector<double> coeffs();
    double predict(const std::vector<double> &x);
    double score(const std::vector<std::vector<double>> &x, const std::vector<double> &y);

    void set_coeffs(const std::vector<double>& coeffs);
    
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

#endif