#ifndef _KFOLD_H_
#define _KFOLD_H_

#include <vector>

class kfold {
public:
    kfold(size_t k, bool shuffle = true);
    ~kfold() = default;

    void split(size_t index, const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::vector<std::vector<double>>& x_train, std::vector<std::vector<double>>& x_test, std::vector<double>& y_train, std::vector<double>& y_test);
    size_t k();

private:
    void prepareIndices(size_t count);

    size_t m_k;
    size_t m_blockSize;
    bool m_shuffle;
    std::vector<size_t> m_indices;
};

#endif