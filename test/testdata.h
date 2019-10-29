#ifndef _TESTDATA_H_
#define _TESTDATA_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class test_data {
public:
    test_data();
    ~test_data();

    static void parse(std::string file, std::vector<int> data_columns, Eigen::MatrixXd& data)
    {
        std::ifstream fis(file);
        std::string line, token;
        bool firstline = true;
        size_t column, index, lines = 0, line_count;

        auto state_backup = fis.rdstate();
        fis.clear();
        auto pos_backup = fis.tellg();
        fis.seekg(0);
        line_count = std::count(std::istreambuf_iterator<char>(fis), std::istreambuf_iterator<char>(), '\n') - 1;
        fis.unget();
        if( fis.get() != '\n' ) { ++line_count ; } 
        // recover state
        fis.clear();
        fis.seekg(pos_backup);
        fis.setstate(state_backup);

        data.conservativeResize(line_count, data_columns.size());

        while (fis.good()) {
            std::getline(fis, line);
            if (firstline) 
                firstline = false;
            else {
                std::istringstream tokenStream(line);
                column = 0;
                index = 0;

                while (std::getline(tokenStream, token, ','))
                {
                    if (std::find(data_columns.begin(), data_columns.end(), column) != data_columns.end())
                        data(lines, index++) = stod(token);
                    column++;
                }
                lines++;
            }
        }
   }
};

#endif