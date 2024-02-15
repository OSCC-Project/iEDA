#include "mt-kahypar/utils/reproducible_random.h"

#include <iostream>
#include <random>
#include <tbb/global_control.h>

namespace mt_kahypar::utils {


void benchShuffle(size_t n, int num_threads) {
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

#ifndef NDEBUG
  auto is_permutation = [&](vec<int>& r1, vec<int>& r2) {
    std::sort(r1.begin(), r1.end());
    std::sort(r2.begin(), r2.end());
    return r1 == r2;
  };
#endif

  ParallelPermutation<int, PrecomputeBucket> shuffle_preassign;
  ParallelPermutation<int, PrecomputeBucketOpt> shuffle_preassign_opt;
  ParallelPermutation<int, BucketHashing> shuffle_hash;

  uint32_t seed = 420;
  std::mt19937 rng(seed);

  vec<int> comp(n);
  std::iota(comp.begin(), comp.end(), 0);

  std::cout << "preassign buckets" << std::endl;
  shuffle_preassign.create_integer_permutation(n, num_threads, rng);
  assert(is_permutation(comp, shuffle_preassign.permutation));

  std::cout << "preassign buckets opt" << std::endl;
  rng.seed(seed);
  shuffle_preassign_opt.create_integer_permutation(n, num_threads, rng);
  assert(is_permutation(comp, shuffle_preassign_opt.permutation));

  std::cout << "hash" << std::endl;
  rng.seed(seed);
  shuffle_hash.create_integer_permutation(n, num_threads, rng);
  assert(is_permutation(comp, shuffle_hash.permutation));
}

void testGroupingReproducibility(size_t n, int num_threads) {
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, num_threads);

  size_t num_reps = 5;
  using Permutation = ParallelPermutation<int, PrecomputeBucketOpt>;

  Permutation first;
  for (size_t i = 0; i < num_reps; ++i) {
    Permutation shuffle;
    shuffle.random_grouping(n, num_threads, 420);

    if (i == 0) {
      first = shuffle;
    } else {
      assert(shuffle.get_bucket.precomputed_buckets == first.get_bucket.precomputed_buckets);
      assert(shuffle.permutation == first.permutation);
      assert(shuffle.bucket_bounds == first.bucket_bounds);
    }
  }
}


void testFeistel() {
  std::mt19937 rng(420);

  size_t max_num_entries = UL(1) << 62;
  FeistelPermutation feistel;
  feistel.create_permutation(251, max_num_entries, rng);

  auto t = [&](uint64_t plain_text) {
    uint64_t encrypted = feistel.encrypt(plain_text);
    uint64_t decrypted = feistel.decrypt(encrypted);
    assert(decrypted == plain_text);
  };

  t(420);
  t(245252);
  t(11);
  t(max_num_entries - 1);

  for (size_t i = 0; i < 500; ++i) {
    t(i);
  }
}


}


int main(int argc, char* argv[]) {

  if (argc != 3) {
    std::cout << "Usage. num-threads permutation-size" << std::endl;
    std::exit(0);
  }

  int num_threads = std::stoi(argv[1]);
  size_t n = std::stoi(argv[2]);
  // mt_kahypar::utils::benchShuffle(n, num_threads);
  mt_kahypar::utils::testGroupingReproducibility(n, num_threads);

  // mt_kahypar::utils::testFeistel();
  return 0;
}
