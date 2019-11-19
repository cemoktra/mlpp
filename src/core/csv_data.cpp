#include "csv_data.h"
#include "csv_reader.h"
#include <stdexcept>
#include <iostream>

void csv_data::read(const std::string& file)
{
    csv_reader csv (nullptr, std::bind(&csv_data::add_row, this, std::placeholders::_1, std::placeholders::_2), std::bind(&csv_data::add_header, this, std::placeholders::_1));
    m_data.clear();
    m_headers.clear();
    csv.read(file);
}

void csv_data::add_row(size_t row, std::list<std::string> tokens)
{
    if (!m_data.size())
        m_data.resize(tokens.size());
    if (m_data.size() != tokens.size()) {
        std::cout << "row " << row << ": " << m_data.size() << " vs " << tokens.size() << std::endl;
        throw std::invalid_argument("rows have different number of columns");
    }
    
    size_t index = 0;
    for (auto token : tokens)
        m_data[index++].push_back(token);
}

void csv_data::add_header(std::list<std::string> tokens)
{
    m_headers = tokens;
}

size_t csv_data::rows() const 
{
    if (!m_data.size())
        return 0;
    return m_data[0].size();
}

size_t csv_data::cols() const
{
    return m_data.size();
}

std::list<std::string> csv_data::headers()
{
    return m_headers;
}

template<>
std::vector<std::string> csv_data::col(size_t index) const
{
    return m_data[index];
}

template<>
std::vector<double> csv_data::col(size_t index) const
{
    std::vector<double> result;
    if (m_stringConversion == UniqueStringIndex) {
        std::vector<std::string> dictionary;
        std::unique_copy(m_data[index].begin(), m_data[index].end(), std::back_inserter(dictionary));
        std::for_each(m_data[index].begin(), m_data[index].end(), [&](const std::string& value) { 
            auto dictionary_it = std::find(dictionary.begin(), dictionary.end(), value);
            if (dictionary_it == dictionary.end())
                throw std::exception();
            result.push_back(dictionary_it - dictionary.begin()); 
        });
    } else {
        std::for_each(m_data[index].begin(), m_data[index].end(), [&](const std::string& value) { 
            result.push_back(stod(value)); 
        });
    }
    return result;
}

Eigen::MatrixXd csv_data::matrixFromCols(std::vector<size_t> cols, EStringToDoubleTypes conversion)
{
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(rows(), cols.size());
    m_stringConversion = conversion;
    size_t index = 0;
    for (auto c : cols)
        matrix.col(index++) = Eigen::Map<Eigen::VectorXd>(this->col<double>(c).data(), matrix.rows());
    return matrix;
}
