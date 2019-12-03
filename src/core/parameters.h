#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include <string>
#include <map>

class parameters
{
public:
    parameters() = default;
    parameters(const parameters&) = delete;
    ~parameters() = default;
    
    double get_param(const std::string& name) const;
    virtual void set_param(const std::string& name, double new_value);

    void copy_params(const parameters& rhs);
    
protected:
    void register_param(const std::string& name, double default_value);

private:
    std::map<std::string, double> m_parameters;
};

#endif