#ifndef _CSV_READER_H_
#define _CSV_READER_H_

#include <string>
#include <fstream>
#include <functional>

typedef std::function<void(size_t)> LineCountCB;
typedef std::function<void(size_t, std::vector<std::string>)> LineCB;

class csv_reader
{
public:
    csv_reader(LineCountCB lineCountCB, LineCB lineCB);
    csv_reader(const csv_reader&) = delete;
    ~csv_reader() = default;

    void read(const std::string& file);

private:
    size_t count_lines(std::ifstream& fs);
    void parse_line(const std::string& line);

    LineCountCB m_lineCountCB;
    LineCB      m_lineCB;
    char        m_delim;
    size_t      m_line_index;
};

#endif