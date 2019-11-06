#include "vocabulary.h"
#include <iostream>

void vocabulary::add(const std::vector<std::string>& data)
{
    for (auto& item : data) {
        for (auto i = strtok(strdup(item.c_str()), " "); i != nullptr; i = strtok(nullptr, " ")) {
            // TODO: remove trailing chars which are not part of the word (e.g. currently the word may be "point,")
            if (m_vocabulary.find(i) == m_vocabulary.end()) 
                m_vocabulary[i] = 1;
            else
                m_vocabulary[i]++;
        }
    }
}

Eigen::SparseMatrix<double> vocabulary::transform(const std::vector<std::string>& data)
{
    Eigen::SparseMatrix<double> result(data.size(), m_vocabulary.size());
    std::vector<Eigen::Triplet<double>> triplets;

    for (auto i = 0; i < data.size(); i++)
    {
        for (auto j = strtok(strdup(data[i].c_str()), " "); j != nullptr; j = strtok(nullptr, " ")) {
            auto it = m_vocabulary.find(j);
            if (it != m_vocabulary.end())
                triplets.push_back(Eigen::Triplet<double>(i, std::distance(m_vocabulary.begin(), it), it->second));
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}