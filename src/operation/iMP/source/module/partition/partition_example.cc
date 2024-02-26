#include <cassert>
#include <random>
#include <ranges>
#include <thread>

#include "Hmetis.hh"
#include "MtKahypar.hh"
int main()
{
  size_t N = 10000;
  size_t E = 10000;
  size_t min_degree = 2;
  int seed = std::random_device()();  // set random seed
  std::mt19937 gen1(seed);
  std::mt19937 gen2(seed);
  std::mt19937 gen3(seed);
  std::poisson_distribution<size_t> rand_degree(min_degree);
  std::uniform_int_distribution<size_t> dis(0, N - 1);
  std::uniform_real_distribution<double> real(0.001, 0.01);
  // Random genertate net
  std::vector<size_t> eptr(E + 1);
  size_t cur = 0;
  auto get_degree = [&]() {
    auto size = rand_degree(gen1);
    return size > 1 ? size : 2;
  };
  std::generate(eptr.begin(), eptr.end(), [&]() { return cur += get_degree(); });
  std::vector<size_t> eind(eptr.back());
  std::generate(eind.begin(), eind.end(), [&]() { return dis(gen2); });
  std::vector<int32_t> net_weight(E, 1);
  imp::MtKahypar mt_kahypartition{.num_threads = 16};
  // imp::MtKahypar mt_kahypartition{.num_threads = 1};

  double nparts_t = N * real(gen3);

  size_t nparts = nparts_t > 2. ? std::round(nparts_t) : 2;

  auto part = mt_kahypartition("test", eptr, eind, nparts);
  // size_t max_part = *std::max_element(part.begin(), part.end());
  // assert(max_part==)
  return 0;
}