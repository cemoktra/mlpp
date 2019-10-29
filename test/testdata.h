#ifndef _TESTDATA_H_
#define _TESTDATA_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>

class test_data {
public:
    test_data();
    ~test_data();

    static void parse(std::string file, std::vector<int> data_columns, Eigen::MatrixXd& data, const std::map<std::string, double>& class_map = std::map<std::string, double>())
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
                    if (std::find(data_columns.begin(), data_columns.end(), column) != data_columns.end()) {
                        try {
                            data(lines, index++) = stod(token);
                        } catch (...) {
                            auto mapped_value = class_map.find(token);
                            if (mapped_value != class_map.end()) {
                                data(lines, index++) = mapped_value->second;
                            } else {
                                data(lines, index++) = std::numeric_limits<double>::quiet_NaN();
                                std::cout << "Could not handle csv value: " << token << std::endl;
                            }
                        }
                    }
                    column++;
                }
                lines++;
            }
        }
   }
};

#endif 