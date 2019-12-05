#include <benchmark/benchmark.h>
#include <core/normalize.h>
#include <core/one_hot.h>
#include <core/standard_scale.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

static const size_t matrix_size = 1000;

static void BM_Normalize(benchmark::State& state) {
  xt::xarray<double> test_data = xt::random::rand<double>(std::vector<size_t>({matrix_size, matrix_size}));

  normalize scaler;
  for (auto _ : state) {
    scaler.fit(test_data);
    auto result = scaler.transform(test_data);
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_StandardScale(benchmark::State& state) {
  xt::xarray<double> test_data = xt::random::rand<double>(std::vector<size_t>({matrix_size, matrix_size}));

  standard_scale scaler;
  for (auto _ : state) {
    scaler.fit(test_data);
    auto result = scaler.transform(test_data);
  }

  state.SetItemsProcessed(state.iterations());
}

static void BM_One_Hot(benchmark::State& state) {
  xt::xarray<int> test_data = xt::random::randint<int>(std::vector<size_t>({matrix_size, 1}), 0, 10);
  for (auto _ : state) {
    auto result = one_hot::transform(test_data);
  }

  state.SetItemsProcessed(state.iterations());
}


BENCHMARK(BM_Normalize);
BENCHMARK(BM_StandardScale);
BENCHMARK(BM_One_Hot);

BENCHMARK_MAIN();