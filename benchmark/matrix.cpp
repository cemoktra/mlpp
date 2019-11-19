#include <benchmark/benchmark.h>
#include <core/matrix.h>
#include <Eigen/Dense>

static const size_t matrix_size = 1000;

static void BM_MatrixAdd(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, matrix_size);
  m1 = 1.0;
  m2 = 1.0;

  for (auto _ : state) {
    m1 += m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixAddCol(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, 1);
  m1 = 1.0;
  m2 = 1.0;

  for (auto _ : state) {
    m1 += m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixAddRow(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (1, matrix_size);
  m1 = 1.0;
  m2 = 1.0;

  for (auto _ : state) {
    m1 += m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixExp(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  m1 = 1.0;

  for (auto _ : state) {
    m1.exp();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixMatMul(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, matrix_size);
  m1 = 1.0;
  m2 = 1.0;

  for (auto _ : state) {
    m1 = m1.matmul(m2);
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_MatrixTranspose(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  m1 = 1.0;

  for (auto _ : state) {
    auto m2 = m1.transpose();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenAdd(benchmark::State& state) {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);

  for (auto _ : state) {
    m1.array() += m2.array();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenAddCol(benchmark::State& state) {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Ones(matrix_size, 1);

  for (auto _ : state) {
    m1.array().colwise() += m2.col(0).array();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenAddRow(benchmark::State& state) {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Ones(1, matrix_size);

  for (auto _ : state) {
    m1.array().rowwise() += m2.row(0).array();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenExp(benchmark::State& state) {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);

  for (auto _ : state) {
    m1 = m1.array().exp();
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenMatMul(benchmark::State& state) {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);
  Eigen::MatrixXd m2 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);

  for (auto _ : state) {
    m1 = m1 * m2;
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_EigenTranspose(benchmark::State& state) {
  Eigen::MatrixXd m1 = Eigen::MatrixXd::Ones(matrix_size, matrix_size);

  for (auto _ : state) {
    Eigen::MatrixXd m2 = m1.transpose();
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_MatrixAdd);
BENCHMARK(BM_MatrixAddCol);
BENCHMARK(BM_MatrixAddRow);
BENCHMARK(BM_MatrixExp);
BENCHMARK(BM_MatrixMatMul);
BENCHMARK(BM_MatrixTranspose);

BENCHMARK(BM_EigenAdd);
BENCHMARK(BM_EigenAddCol);
BENCHMARK(BM_EigenAddRow);
BENCHMARK(BM_EigenExp);
BENCHMARK(BM_EigenMatMul);
BENCHMARK(BM_EigenTranspose);

BENCHMARK_MAIN();