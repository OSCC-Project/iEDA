#include "Partitionner.hh"

#include <time.h>

#include <cassert>
#include <unordered_map>

#include "Logger.hpp"
const std::string khmetis_binary_path = HMETIS_BINARY;

namespace imp {
vector<size_t> Partitionner::hmetisSolve(size_t num_vertexs, size_t num_hedges, const vector<size_t>& eptr, const vector<size_t>& eind,
                                         size_t nparts, float ufactor, const vector<int64_t>& vwgt, const vector<int64_t>& hewgt,
                                         int n_runs, int seed)
{
  assert(vwgt.empty() || vwgt.size() == num_vertexs);
  assert(hewgt.empty() || hewgt.size() == num_hedges);
  vector<size_t> parts;
  std::string curr_time = std::to_string(int(clock()));
  std::string hgraph_file_name = "/tmp/ieda_imp_hmetis_input" + curr_time + ".hgr";
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
    for (size_t j = eptr[i]; j < eptr[i + 1]; j++) {
      hgraph_file << eind[j] + 1 << " ";
    }
    hgraph_file << std::endl;
  }
  for (int64_t var : vwgt) {
    hgraph_file << var << std::endl;
  }
  hgraph_file.close();

  std::string cmd = khmetis_binary_path + " " + hgraph_file_name + " " + std::to_string(nparts);
  //  " " + std::to_string(ufactor) + " 10 5 3 3 0 0";
  std::string ptype = "rb";
  std::string ctype = "gfc1";
  std::string rtype = "moderate";
  std::string otype = "cut";
  int dbglvl = 0;
  int seed = 0;
  bool reconst = false;

  cmd += " -ptype=" + ptype;
  cmd += " -ctype=" + ctype;
  cmd += " -rtype=" + rtype;
  cmd += " -otype=" + otype;
  cmd += " -ufactor=" + std::to_string(ufactor);
  cmd += " -nruns=" + std::to_string(n_runs);
  cmd += " -dbglvl=" + std::to_string(dbglvl);
  cmd += " -seed=" + std::to_string(seed);
  if (reconst) {
    cmd += " -reconst";
  }

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
#ifdef NDEBUG
  std::system(rm hgraph_file_name);
  std::system(rm solution_file);
#endif

  return parts;
}
}  // namespace imp
