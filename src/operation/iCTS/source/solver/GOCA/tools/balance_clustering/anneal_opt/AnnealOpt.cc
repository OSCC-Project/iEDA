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
/**
 * @file AnnealOpt.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "AnnealOpt.hh"

#include <algorithm>
#include <cmath>

#include "TimingPropagator.hh"
#include "log/Log.hh"
namespace icts {
/**
 * @brief init the AnnealOpt solver parameters
 *
 * @param net_num
 * @param max_fanout
 * @param max_cap
 * @param max_net_dist
 * @param p
 * @param q
 * @param r
 */
void AnnealOpt::initParameter(const size_t& net_num, const int& max_fanout, const double& max_cap, const int& max_net_dist, const double& p,
                              const double& q, const double& r)
{
  _net_num = net_num;
  _max_fanout = max_fanout;
  _max_cap = max_cap;
  _max_net_dist = max_net_dist;
  _p = p;
  _q = q;
  _r = r;

  _inst_num = _flatten_insts.size();

  std::ranges::for_each(_flatten_insts, [&](const Inst* inst) {
    auto loc = inst->get_location();
    _min_x = std::min(_min_x, loc.x());
    _min_y = std::min(_min_y, loc.y());
    _max_x = std::max(_max_x, loc.x());
    _max_y = std::max(_max_y, loc.y());
  });
}

std::vector<std::vector<Inst*>> AnnealOpt::run()
{
  return std::vector<std::vector<Inst*>>();
}

}  // namespace icts