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

TEST_F (matrix_test, add_col) { 
    matrix m1(3, 3), m2(3, 1);

    int cnt = 0;
    for (auto it = m1.begin(); it != m1.end(); ++it, ++cnt)
        *it = cnt;
    cnt = 0;
    for (auto it = m2.begin(); it != m2.end(); ++it, ++cnt)
        *it = cnt;

    auto m3 = m1 + m2;

    for (auto i = 0; i < 9; i++)
        ASSERT_EQ(m3.get_at(i / 3, i % 3), i + i / 3);
}

TEST_F (matrix_test, add_row) { 
    matrix m1(3, 3), m2(1, 3);

    int cnt = 0;
    for (auto it = m1.begin(); it != m1.end(); ++it, ++cnt)
        *it = cnt;
    cnt = 0;
    for (auto it = m2.begin(); it != m2.end(); ++it, ++cnt)
        *it = cnt;

    auto m3 = m1 + m2;

    for (auto i = 0; i < 9; i++)
        ASSERT_EQ(m3.get_at(i / 3, i % 3), i + i % 3);
}

TEST_F (matrix_test, exp) { 
    matrix m1(3, 3);

    int cnt = 0;
    for (auto it = m1.begin(); it != m1.end(); ++it, ++cnt)
        *it = cnt;

    m1.exp();
    cnt = 0;
    for (auto it = m1.begin(); it != m1.end(); ++it, ++cnt)
        ASSERT_NEAR(*it, std::exp(cnt), 1e-6);
}

TEST_F (matrix_test, transpose) { 
    matrix m1(3, 3);

    int cnt = 0;
    for (auto it = m1.begin(); it != m1.end(); ++it, ++cnt)
        *it = cnt;

    auto m2 = m1.transpose();
    
    cnt = 0;
    for (auto it = m1.col_begin(0); it != m1.col_end(0); ++it, ++cnt)
        ASSERT_EQ(*it, cnt);
}

TEST_F (matrix_test, mat_mul) { 
    matrix m1(2, 3);
    matrix m2(3, 2);

    m1 = 0.0;
    m1.set_at(0, 0, 3);
    m1.set_at(0, 1, 2);
    m1.set_at(0, 2, 1);
    m1.set_at(1, 0, 1);
    m1.set_at(1, 2, 2);

    m2 = 0.0;
    m2.set_at(0, 0, 1);
    m2.set_at(2, 0, 4);
    m2.set_at(0, 1, 2);
    m2.set_at(1, 1, 1);

    m1 = m1.matmul(m2);

    ASSERT_EQ(m1.rows(), 2);
    ASSERT_EQ(m1.cols(), 2);
    ASSERT_EQ(m1.get_at(0, 0), 7.0);
    ASSERT_EQ(m1.get_at(0, 1), 8.0);
    ASSERT_EQ(m1.get_at(1, 0), 9.0);
    ASSERT_EQ(m1.get_at(1, 1), 2.0);
}