#include "vocabulary.h"
#include <iostream>

void vocabulary::add(const std::vector<std::string>& data)
{
    for (auto& item : data) {
        for (auto i = strtok(strdup(item.c_str()), " "); i != nullptr; i = strtok(nullptr, " ")) {
            std::string word (i);
            std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c){ return std::tolower(c); });
            word = word.substr(0, word.find_first_of(','));
            word = word.substr(0, word.find_first_of('.'));

            if (m_vocabulary.find(word) == m_vocabulary.end()) 
                m_vocabulary[word] = 1;
            else
                m_vocabulary[word]++;
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
            std::string word (j);
            std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c){ return std::tolower(c); });
            word = word.substr(0, word.find_first_of(','));
            word = word.substr(0, word.find_first_of('.'));

            auto it = m_vocabulary.find(word);
            if (it != m_vocabulary.end())
                triplets.push_back(Eigen::Triplet<double>(i, std::distance(m_vocabulary.begin(), it), 1));
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    
    return result;
}

void vocabulary::filter_counts(size_t min_occurences, size_t max_occurences)
{
    auto it = m_vocabulary.begin();
    while (it != m_vocabulary.end())
    {
        if (it->second < min_occurences)
            it = m_vocabulary.erase(it);
        else if (max_occurences > 0 && it->second > max_occurences)
            it = m_vocabulary.erase(it);
        else
            it++;
    }
}

void vocabulary::filter_perc(double min_occurences, double max_occurences)
{
    size_t words = m_vocabulary.size();
    size_t min_occ = ceil(min_occurences * words);
    size_t max_occ = max_occurences > 0.0 ? ceil(max_occurences * words) : 0;
    filter_counts(min_occ, max_occ);
}