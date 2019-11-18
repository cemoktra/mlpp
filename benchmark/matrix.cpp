#include <benchmark/benchmark.h>
#include <core/matrix.h>
#include <Eigen/Dense>

static const size_t matrix_size = 1000;

static void BM_MatrixAdd(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, matrix_size);

  for (auto _ : state) {
    m1 += m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixAddCol(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, 1);

  for (auto _ : state) {
    m1 += m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixMul(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, matrix_size);

  for (auto _ : state) {
    m1 *= m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixMulCol(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, 1);

  for (auto _ : state) {
    m1 *= m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixAddAVX(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, matrix_size);

  for (auto _ : state) {
    m1.avx_add(m2);
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixMulAVX(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, matrix_size);

  for (auto _ : state) {
    m1.avx_mul(m2);
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenAdd(benchmark::State& state) {
  Eigen::MatrixXd m1 (matrix_size, matrix_size);
  Eigen::MatrixXd m2 (matrix_size, matrix_size);

  for (auto _ : state) {
    m1.array() += m2.array();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenAddCol(benchmark::State& state) {
  Eigen::MatrixXd m1 (matrix_size, matrix_size);
  Eigen::MatrixXd m2 (matrix_size, 1);

  for (auto _ : state) {
    m1.array().colwise() += m2.col(0).array();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenMul(benchmark::State& state) {
  Eigen::MatrixXd m1 (matrix_size, matrix_size);
  Eigen::MatrixXd m2 (matrix_size, matrix_size);

  for (auto _ : state) {
    m1.array() *= m2.array();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenMulCol(benchmark::State& state) {
  Eigen::MatrixXd m1 (matrix_size, matrix_size);
  Eigen::MatrixXd m2 (matrix_size, 1);

  for (auto _ : state) {
    m1.array().colwise() *= m2.col(0).array();
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_MatrixAdd);
BENCHMARK(BM_MatrixAddCol);
BENCHMARK(BM_MatrixAddAVX);
BENCHMARK(BM_MatrixMul);
BENCHMARK(BM_MatrixMulCol);
BENCHMARK(BM_MatrixMulAVX);

BENCHMARK(BM_EigenAdd);
BENCHMARK(BM_EigenAddCol);
BENCHMARK(BM_EigenMul);
BENCHMARK(BM_EigenMulCol);

BENCHMARK_MAIN();