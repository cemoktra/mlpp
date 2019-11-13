#include <benchmark/benchmark.h>
#include <core/matrix.h>

static const size_t matrix_size = 1000;

static void BM_MatrixAdd(benchmark::State& state) {
  matrix m1 (matrix_size, matrix_size);
  matrix m2 (matrix_size, matrix_size);

  for (auto _ : state) {
    m1 += m2;
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

BENCHMARK(BM_MatrixAdd);
BENCHMARK(BM_MatrixAddAVX);

BENCHMARK_MAIN();