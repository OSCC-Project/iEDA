#pragma once
#include <cstddef>
#include <map>
#include <memory>
#include <numeric>
#include <unordered_map>

namespace imp {
class Block;
class HMetis;
class ParserEngine;
class Instance;

struct BlkClustering
{
  void operator()(imp::Block& block);

  size_t l1_nparts{std::numeric_limits<size_t>::max()};
  size_t l2_nparts{std::numeric_limits<size_t>::max()};
};

// multilevel-level clustering operation
struct BlkClustering2
{
  void operator()(imp::Block& block);
  void multiLevelClustering(imp::Block& root_cluster);
  void singleLevelClustering(imp::Block& root_cluster);
  size_t l1_nparts{std::numeric_limits<size_t>::max()};
  size_t l2_nparts{std::numeric_limits<size_t>::max()};
  size_t level_num = 2;
  std::weak_ptr<ParserEngine> parser;

  void paramCheck();
};

}  // namespace imp
