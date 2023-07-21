#include "Partitionner.hh"

#include "Logger.hpp"

namespace imp {
vector<size_t> Partitionner::hmetisSolve(size_t num_vertexs, size_t num_hedges, const vector<size_t>& eptr, const vector<size_t>& eind,
                                         size_t nparts, size_t ufactor, const vector<int32_t>& vwgt, const vector<int32_t>& hewgt)
{
  vector<size_t> parts;
  std::string hgraph_file_name = "./input.txt";
  std::ofstream hgraph_file(hgraph_file_name);
  hgraph_file << num_hedges << " " << num_vertexs << std::endl;
  for (size_t i = 0; i < num_hedges; i++) {
    for (size_t j = eptr[i]; j < eptr[i + 1] - 1; j++) {
      hgraph_file << eind[j] << " ";
    }
    hgraph_file << eind[eptr[i + 1] - 1] << std::endl;
  }
  hgraph_file.close();

  std::string cmd = khmetis_binary_path + " " + hgraph_file_name + " " + std::to_string(nparts) + std::to_string(ufactor) + " 10 5 3 3 0 0";
  INFO("Starting hmetis partition ...");
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
