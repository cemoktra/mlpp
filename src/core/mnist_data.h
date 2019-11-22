#ifndef _MNIST_DATA_H_
#define _MNIST_DATA_H_

#include <string>
#include <zlib.h>
#include <xtensor/xarray.hpp>

class mnist_data
{
public:
    mnist_data() = default;
    mnist_data(const mnist_data&) = delete;
    ~mnist_data() = default;

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> read(const std::string& folder, const std::string& train_data_file, const std::string& train_label_file, const std::string& test_data_file, const std::string& test_label_file);

private:
    xt::xarray<double> read_file(const std::string& folder, const std::string& data_file);

    size_t read_dimensions(gzFile gzf);

    uint32_t convert(uint32_t value);
};

#endif