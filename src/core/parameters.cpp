#include "parameters.h"
#include <stdexcept>

double parameters::get_param(const std::string& name)
{
    auto it = m_parameters.find(name);
    if (it == m_parameters.end())
        throw std::invalid_argument("unknown parameter");
    else
        return it->second;
}

void parameters::set_param(const std::string& name, double new_value)
{
    auto it = m_parameters.find(name);
    if (it == m_parameters.end())
        throw std::invalid_argument("unknown parameter");
    else
        it->second = new_value;
}

void parameters::register_param(const std::string& name, double default_value)
{
    m_parameters[name] = default_value;
}