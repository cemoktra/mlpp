#ifndef _CSV_DATA_H_
#define _CSV_DATA_H_

#include <list>
#include <string>
#include <Eigen/Dense>

class csv_data {
public:
    enum EStringToDoubleTypes {
        ParseValue,
        UniqueStringIndex
    };

    csv_data() = default;
    csv_data(const csv_data&) = delete;
    ~csv_data() = default;

    void read(const std::string& file);

    size_t rows() const;
    size_t cols() const;

    template<typename T>
    std::vector<T> col(size_t index) const;

    Eigen::MatrixXd matrixFromCols(std::vector<size_t> cols, EStringToDoubleTypes conversion = ParseValue);

    std::list<std::string> headers();

private:
    void add_header(std::list<std::string> tokens);
    void add_row(size_t row, std::list<std::string> tokens);

    typedef std::vector<std::string> col_type;
    typedef std::vector<col_type> csv_data_type;
    typedef std::list<std::string> csv_headers;

    EStringToDoubleTypes m_stringConversion = ParseValue;
    csv_data_type m_data;
    csv_headers m_headers;

    size_t m_rows;
};

#endif