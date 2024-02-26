#pragma once
#include <string>
#include <vector>
namespace imp {

struct MtKahypar
{
  std::vector<size_t> operator()(const std::string name, const std::vector<size_t>& eptr, const std::vector<size_t>& eind, size_t nparts,
                                 const std::vector<int>& vwgt = {}, const std::vector<int>& hewgt = {});
  size_t num_threads;
};

}  // namespace imp