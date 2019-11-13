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

TEST_F (matrix_test, add_avx) { 
    matrix m1(3, 3), m2(3, 3);

    for (auto i = 0; i < 9; i++) {
        m1.set_at(i / 3, i % 3, 1);
        m2.set_at(i / 3, i % 3, 2);
    }

    m1.avx_add(m2);
    for (auto i = 0; i < 9; i++)
        ASSERT_EQ(m1.get_at(i / 3, i % 3), 3);
}
