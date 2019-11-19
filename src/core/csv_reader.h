#ifndef _CSV_READER_H_
#define _CSV_READER_H_

#include <string>
#include <fstream>
#include <functional>
#include <list>

typedef std::function<void(size_t)> LineCountCB;
typedef std::function<void(size_t, std::list<std::string>)> LineCB;
typedef std::function<void(std::list<std::string>)> HeaderCB;

class csv_reader
{
public:
    csv_reader(LineCountCB lineCountCB, LineCB lineCB, HeaderCB headerCB = nullptr);
    csv_reader(const csv_reader&) = delete;
    ~csv_reader() = default;

    void read(const std::string& file);

private:
    size_t count_lines(std::ifstream& fs);
    std::list<std::string> parse_line(const std::string& line);

    LineCountCB m_lineCountCB;
    LineCB      m_lineCB;
    HeaderCB    m_headerCB;
    char        m_delim;
    size_t      m_line_index;
};

#endif