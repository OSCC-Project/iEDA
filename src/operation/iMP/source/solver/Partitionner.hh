/**
 * @file Partitionner.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-07-11
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_PARTITIONNER_H
#define IMP_PARTITIONNER_H
#include <fstream>
#include <string>
#include <vector>
using std::vector;
const std::string khmetis_binary_path = "../src/third_party/hmetis/hmetis2.0pre1";
namespace imp {

class Partitionner
{
 public:
  static bool hmetisSolve(int num_vertexs, int num_hedges, const vector<int>& eptr,
                          const vector<int>& eind, vector<int>& parts, int nparts = 500,
                          int ufactor = 5, const vector<int>& vwgt = {},
                          const vector<int>& hewgt = {});

 private:
  Partitionner() = delete;
  ~Partitionner() = delete;
};

}  // namespace imp

#endif