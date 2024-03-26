#pragma once
#include <cstddef>
#include <memory>
#include <numeric>
namespace imp {
class Block;
class HMetis;
class ParserEngine;

struct BlkClustering
{
  void operator()(imp::Block& block);

  size_t l1_nparts{std::numeric_limits<size_t>::max()};
  size_t l2_nparts{std::numeric_limits<size_t>::max()};
};

struct BlkClustering2
{
  void operator()(imp::Block& block);
  size_t l1_nparts = 200;
  size_t l2_nparts = 0;
  size_t level_num = 1;
  std::weak_ptr<ParserEngine> parser;
};

}  // namespace imp
