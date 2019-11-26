#ifndef _SCOPED_TIMER_H_
#define _SCOPED_TIMER_H_

#include <chrono>
#include <iostream>

class scoped_timer
{
public:
    scoped_timer(const std::string& name) 
        : m_start(std::chrono::high_resolution_clock::now())
    {
        std::cout << name << " ... ";
    }

    ~scoped_timer()
    {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::high_resolution_clock::now() - m_start);
        std::cout << "done [" << duration.count() << "ms]" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

#endif