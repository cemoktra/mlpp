#include "csv_reader.h"

csv_reader::csv_reader(LineCountCB lineCountCB, LineCB lineCB)
    : m_lineCountCB(lineCountCB)
    , m_lineCB(lineCB)
    , m_delim(',')
{
    
}

void csv_reader::read(const std::string& file)
{
    std::ifstream fs(file);
    std::string line;
    bool header = true;

    auto lines = count_lines(fs);
    m_line_index = 0;
    if (m_lineCountCB)
        m_lineCountCB(lines);

    while (fs.good()) {
        std::getline(fs, line);
        if (header)
            header = false;
        else
            parse_line(line);
    }
}

size_t csv_reader::count_lines(std::ifstream& fs)
{
    auto state_backup = fs.rdstate();
    auto pos_backup = fs.tellg();
    fs.clear();
    fs.seekg(0);
    size_t line_count = std::count(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>(), '\n') - 1;
    fs.unget();
    if (fs.get() != '\n' ) 
        ++line_count;
    // recover state
    fs.clear();
    fs.seekg(pos_backup);
    fs.setstate(state_backup);
    return line_count;
}

void csv_reader::parse_line(const std::string& line)
{
    std::vector<std::string> tokens;
    std::string token;
    size_t pos = 0;
    while (true) {
        size_t end = std::string::npos;
        auto delim_pos = line.find_first_of(m_delim, pos);
        auto quote_pos = line.find_first_of('\"', pos);
        
        if (quote_pos != std::string::npos && quote_pos < delim_pos) {
            auto quote_end = line.find_first_of('\"', quote_pos + 1);
            delim_pos = line.find_first_of(m_delim, quote_end);
            token = line.substr(quote_pos + 1, quote_end - quote_pos - 1);
        } else
            token = line.substr(pos, delim_pos - pos);
        tokens.push_back(token);
        if (delim_pos == std::string::npos)
            break;
        pos = delim_pos + 1;
        if (pos >= line.size())
            break;        
    }

    if (m_lineCB)
        m_lineCB(m_line_index++, tokens);
}