// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "Hmetis.hh"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#include "Logger.hpp"
namespace imp {
inline const std::string HMetis::khmetis_binary_path= HMETIS_BINARY;
std::vector<size_t> HMetis::operator()(const std::vector<size_t>& eptr, const std::vector<size_t>& eind, size_t nparts,
                                 const std::vector<int>& vwgts, const std::vector<int>& hewgts)
{
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  std::string hgraph_file_name{oss.str() + ".hgr"}; //using thread id for thread safe.
  std::ofstream hgraph_file(hgraph_file_name);

  size_t num_hedges = eptr.size() - 1;
  size_t num_vertices = !vwgts.empty() ? vwgts.size() : *std::max_element(std::begin(eind), std::end(eind)) + size_t(1);

  hgraph_file << num_hedges << " " << num_vertices;
  if (!vwgts.empty() || !hewgts.empty()) {
    hgraph_file << " " << 1;
    if (!vwgts.empty() && hewgts.empty())
      hgraph_file << 0;
    else if (!vwgts.empty() && !hewgts.empty())
      hgraph_file << " " << 1;
  }
  hgraph_file << std::endl;

  for (size_t i = 0; i < num_hedges; i++) {
    if (!hewgts.empty())
      hgraph_file << hewgts[i] << " ";
    for (size_t j = eptr[i]; j < eptr[i + 1]; j++) {
      hgraph_file << eind[j] + 1 << " ";
    }
    hgraph_file << std::endl;
  }

  for (int var : vwgts) {
    hgraph_file << var << std::endl;
  }
  hgraph_file.close();

  // Construct the command
  std::string command = khmetis_binary_path + " " + hgraph_file_name + " " + std::to_string(nparts);
  command += " -ptype=" + ptypes[ptype];
  command += " -ctype=" + ctypes[ctype];
  command += " -rtype=" + rtypes[rtype];
  command += " -otype=" + otypes[otype];
  command += " -ufactor=" + std::to_string(ufactor);
  command += " -nruns=" + std::to_string(nruns);
  command += " -nvcycles=" + std::to_string(nvcycles);
  command += " -cmaxnet=" + std::to_string(cmaxnet);
  command += " -rmaxnet=" + std::to_string(rmaxnet);

  if (reconst) {
    command += " -reconst";
  }

  if (kwayrefine) {
    command += " -kwayrefine";
  }

  command += " -seed=" + std::to_string(seed);
  command += " -dbglvl=" + std::to_string(dbglvl);
  INFO("Starting hmetis partition ...");
  INFO(command);
  int status = system(command.c_str());

  if (-1 != status && WIFEXITED(status) && 0 == WEXITSTATUS(status)) {
    INFO("hmetis partition succeed..");
  } else {
    ERROR("hmetis partition fail, system return ", status);
    return {};
  }

  std::string solution_file = hgraph_file_name + ".part." + std::to_string(nparts);
  std::ifstream result_file(solution_file);
  std::vector<size_t> parts;
  size_t part_id;
  parts.resize(num_vertices, 0);
  for (size_t i = 0; i < num_vertices && result_file >> part_id; i++) {
    parts[i] = part_id;
  }
  result_file.close();

  std::filesystem::remove(hgraph_file_name);
  std::filesystem::remove(solution_file);
  return parts;
}
}  // namespace imp
