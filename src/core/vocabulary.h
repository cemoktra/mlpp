#ifndef _VOCABULARY_H_
#define _VOCABULARY_H_

#include <Eigen/Sparse>
#include <vector>
#include <map>
#include <string>

class vocabulary {
public:
    vocabulary() = default;
    vocabulary(const vocabulary&) = delete;
    ~vocabulary() = default;

    void add(const std::vector<std::string>& data);
    
    void filter_counts(size_t min_occurences = 0, size_t max_occurences = 0);
    void filter_perc(double min_occurences = 0, double max_occurences = 0);

    Eigen::SparseMatrix<double> transform(const std::vector<std::string>& data);

private:
    std::map<std::string, size_t> m_vocabulary;
};

#endif