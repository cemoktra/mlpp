#include "mnist_data.h"

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> mnist_data::read(const std::string& folder, const std::string& train_data_file, const std::string& train_label_file, const std::string& test_data_file, const std::string& test_label_file)
{
    return std::make_tuple(read_file(folder, train_data_file), read_file(folder, test_data_file), read_file(folder, train_label_file), read_file(folder, test_label_file));
}

xt::xarray<double> mnist_data::read_file(const std::string& folder, const std::string& data_file)
{
    auto gzf = gzopen((folder + data_file).c_str(), "rb");
    uint32_t value32;
    uint8_t value8;

    auto dimensions = read_dimensions(gzf);
    std::vector<size_t> dimension_vector;
    for (auto d = 0; d < dimensions; d++) 
    {
        gzread(gzf, &value32, 4);
        dimension_vector.push_back(convert(value32));
    }
    
    xt::xarray<double> data (dimension_vector);
    for (auto it = data.begin(); it != data.end(); it++) {
        gzread(gzf, &value8, 1);
        *it = static_cast<double>(value8);
    }

    gzclose(gzf);
    return data;
}

size_t mnist_data::read_dimensions(gzFile gzf)
{
    static uint8_t buffer[4];
    gzread(gzf, buffer, 4);
    return static_cast<size_t>(buffer[3]);
}

uint32_t mnist_data::convert(uint32_t value)
{
    uint8_t *ptr = (uint8_t*) &value;
    return (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
}