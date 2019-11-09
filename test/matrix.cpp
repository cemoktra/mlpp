#include <core/matrix.h>

#include <gtest/gtest.h>
#include <numeric>
#include <random>

static const size_t test_size = 100;

class matrix_test : public ::testing::Test {
public:
    matrix_test() = default;
    ~matrix_test() = default;
};

TEST_F (matrix_test, add) { 
    matrix m1(3, 3), m2(3, 3);

    for (auto i = 0; i < 9; i++) {
        m1.set_at(i / 3, i % 3, 1);
        m2.set_at(i / 3, i % 3, 2);
    }

    auto m3 = m1 + m2;
    for (auto i = 0; i < 9; i++)
        ASSERT_EQ(m3.get_at(i / 3, i % 3), 3);
}



// class iterator {
//         long num = FROM;
//     public:
//         iterator(long _num = 0) : num(_num) {}
//         iterator& operator++() {num = TO >= FROM ? num + 1: num - 1; return *this;}
//         iterator operator++(int) {iterator retval = *this; ++(*this); return retval;}
//         bool operator==(iterator other) const {return num == other.num;}
//         bool operator!=(iterator other) const {return !(*this == other);}
//         long operator*() {return num;}
//         // iterator traits
//         using difference_type = long;
//         using value_type = long;
//         using pointer = const long*;
//         using reference = const long&;
//         using iterator_category = std::forward_iterator_tag;
//     };
//     iterator begin() {return FROM;}
//     iterator end() {return TO >= FROM? TO+1 : TO-1;}