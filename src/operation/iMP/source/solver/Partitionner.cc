#include "Partitionner.hh"

#include <cassert>

#include "Logger.hpp"

namespace imp {
vector<size_t> Partitionner::hmetisSolve(size_t num_vertexs, size_t num_hedges, const vector<size_t>& eptr, const vector<size_t>& eind,
                                         size_t nparts, size_t ufactor, const vector<int64_t>& vwgt, const vector<int64_t>& hewgt)
{
  assert(vwgt.empty() || vwgt.size() == num_vertexs);
  assert(hewgt.empty() || hewgt.size() == num_hedges);
  vector<size_t> parts;
  std::string hgraph_file_name = "./input.hgr";
  std::ofstream hgraph_file(hgraph_file_name);
  hgraph_file << num_hedges << " " << num_vertexs;
  if (!vwgt.empty() || !hewgt.empty()) {
    hgraph_file << " " << 1;
    if (!vwgt.empty() && hewgt.empty())
      hgraph_file << 0;
    else if (!vwgt.empty() && !hewgt.empty())
      hgraph_file << " " << 1;
  }
  hgraph_file << std::endl;

  for (size_t i = 0; i < num_hedges; i++) {
    if (!hewgt.empty())
      hgraph_file << hewgt[i] << " ";
    for (size_t j = eptr[i]; j < eptr[i + 1] - 1; j++) {
      hgraph_file << eind[j] + 1 << " ";
    }
    hgraph_file << eind[eptr[i + 1] - 1] + 1 << std::endl;
  }
  for (int64_t var : vwgt) {
    hgraph_file << var << std::endl;
  }
  hgraph_file.close();

  std::string cmd = khmetis_binary_path + " " + hgraph_file_name + " " + std::to_string(nparts);
  //  " " + std::to_string(ufactor) + " 10 5 3 3 0 0";
  INFO("Starting hmetis partition ...");
  INFO(cmd);
  int status = system(cmd.c_str());

  if (-1 != status && WIFEXITED(status) && 0 == WEXITSTATUS(status)) {
    INFO("hmetis partition succeed..");
  } else {
    ERROR("hmetis partition fail, system return ", status);
    return {};
  }
  std::string solution_file = hgraph_file_name + ".part." + std::to_string(nparts);
  std::ifstream result_file(solution_file);
  size_t part_id;
  parts.resize(num_vertexs, -1);
  for (size_t i = 0; i < num_vertexs && result_file >> part_id; i++) {
    parts[i] = part_id;
  }
  result_file.close();
  return parts;
}
}  // namespace imp
