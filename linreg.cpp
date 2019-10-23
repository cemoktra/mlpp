#include "linreg.h"

#include <math.h>
#include <numeric>

linear_regression::linear_regression()
    : m_rate(0.0002)
    , m_threshold(0.0001)
{
}

void linear_regression::init_coeffs(size_t count)
{
    m_coeffs.resize(count);
    m_coeffs_delta.resize(count);
    std::fill(m_coeffs.begin(), m_coeffs.end(), 1.0);
}

void linear_regression::init_buffers(size_t count)
{
    m_sx.resize(count);
    m_sxx.resize(count);
    m_sxy.resize(count);
}

void linear_regression::train(const std::vector<std::vector<double>> &x, const std::vector<double> &y)
{   
    bool thresholdReached = false;

    if (x.size() == 0)
        return;

    init_coeffs(x.size() + 1);
    init_buffers(x.size());

    while (!thresholdReached) {
        cost(x, y);

        thresholdReached = true;
        for (auto i = 0; i < m_coeffs_delta.size(); ++i) {
            m_coeffs[i] -= m_rate * m_coeffs_delta[i];
            thresholdReached = thresholdReached && fabs(m_coeffs_delta[i]) < m_threshold;
        }
    }
}

std::vector<double> linear_regression::coeffs()
{
    return m_coeffs;
}

double linear_regression::predict(const std::vector<double> &x)
{
    if (x.size() + 1 != m_coeffs.size())
        return std::numeric_limits<double>::quiet_NaN();

    double result = m_coeffs[x.size()];
    for (auto i = 0; i < x.size(); i++)
        result += x[i] * m_coeffs[i];
    return result;
}

double linear_regression::cost(const std::vector<std::vector<double>> &x, const std::vector<double> &y)
{
    size_t samples = y.size();
    size_t lastCoeff = x.size();

    double sy = std::accumulate(y.begin(), y.end(), 0);
    double syy = std::inner_product(y.begin(), y.end(), y.begin(), 0);

    for (auto i = 0; i < x.size(); ++i)  {
        m_sx[i] = std::accumulate(x[i].begin(), x[i].end(), 0);
        m_sxx[i] = std::inner_product(x[i].begin(), x[i].end(), x[i].begin(), 0);
        m_sxy[i] = std::inner_product(x[i].begin(), x[i].end(), y.begin(), 0);
    }

    
    double c = syy - 2 * m_coeffs[lastCoeff] * sy + m_coeffs[lastCoeff] * m_coeffs[lastCoeff] * samples;
    for (auto i = 0; i < lastCoeff; ++i)  {
        c -= 2 * m_coeffs[i] * m_sxy[i];
        c += m_coeffs[i] * m_coeffs[i] * m_sxx[i];
        c += 2 * m_coeffs[i] * m_coeffs[lastCoeff] * m_sx[i];

        for (auto j = i + 1; j < m_coeffs.size(); ++j)
            c += 2 * m_coeffs[i] * m_sx[i] * m_coeffs[j] * m_sx[j];
    }
    c /= samples;

    m_coeffs_delta[lastCoeff] = -sy + m_coeffs[lastCoeff] * samples;
    for (auto i = 0; i < lastCoeff; ++i)  {
        m_coeffs_delta[lastCoeff] += m_coeffs[i] * m_sx[i];

        m_coeffs_delta[i] = -m_sxy[i] + m_coeffs[i] * m_sxx[i] + m_sx[i] * m_coeffs[lastCoeff];
        for (auto j = 0; j < lastCoeff; ++j) {
            if (i == j)
                continue;
            m_coeffs_delta[i] += m_sx[i] * m_coeffs[j] * m_sx[j];
        }
    }
    for (auto i = 0; i < m_coeffs.size(); ++i)
        m_coeffs_delta[i] *= (2.0 / samples);

    return c;
}


