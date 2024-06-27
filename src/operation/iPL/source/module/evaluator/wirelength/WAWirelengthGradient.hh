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
/*
 * @Author: S.J Chen
 * @Date: 2022-03-08 22:18:44
 * @LastEditTime: 2022-11-23 11:58:14
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/wirelength/WAWirelengthGradient.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_EVALUATOR_WA_WIRELENGTH_GRADIENT_H
#define IPL_EVALUATOR_WA_WIRELENGTH_GRADIENT_H

#include <vector>

#include "WirelengthGradient.hh"
#include "data/Rectangle.hh"

namespace ipl {

struct WAPinInfo
{
  WAPinInfo();
  WAPinInfo(const WAPinInfo&) = default;
  WAPinInfo(WAPinInfo&&) = default;

  WAPinInfo& operator=(const WAPinInfo&) = delete;
  WAPinInfo& operator=(WAPinInfo&& other)
  {
    max_ExpSum_x = other.max_ExpSum_x;
    max_ExpSum_y = other.max_ExpSum_y;
    min_ExpSum_x = other.min_ExpSum_x;
    min_ExpSum_y = other.min_ExpSum_y;
    has_MaxExpSum_x = other.has_MaxExpSum_x;
    has_MaxExpSum_y = other.has_MaxExpSum_y;
    has_MinExpSum_x = other.has_MinExpSum_x;
    has_MinExpSum_y = other.has_MinExpSum_y;

    return (*this);
  }

  void reset();

  float max_ExpSum_x = 0;
  float max_ExpSum_y = 0;
  float min_ExpSum_x = 0;
  float min_ExpSum_y = 0;

  unsigned char has_MaxExpSum_x : 1;
  unsigned char has_MaxExpSum_y : 1;

  unsigned char has_MinExpSum_x : 1;
  unsigned char has_MinExpSum_y : 1;
};
inline WAPinInfo::WAPinInfo() : has_MaxExpSum_x(0), has_MaxExpSum_y(0), has_MinExpSum_x(0), has_MinExpSum_y(0)
{
}

inline void WAPinInfo::reset()
{
  max_ExpSum_x = 0;
  max_ExpSum_y = 0;
  min_ExpSum_x = 0;
  min_ExpSum_y = 0;
  has_MaxExpSum_x = 0;
  has_MaxExpSum_y = 0;
  has_MinExpSum_x = 0;
  has_MinExpSum_y = 0;
}

struct WANetInfo
{
  WANetInfo() = default;
  WANetInfo(const WANetInfo&) = default;
  WANetInfo(WANetInfo&&) = default;

  WANetInfo& operator=(const WANetInfo&) = delete;
  WANetInfo& operator=(WANetInfo&& other)
  {
    wa_ExpMinSum_x = other.wa_ExpMinSum_x;
    wa_X_ExpMinSum_x = other.wa_X_ExpMinSum_x;
    wa_ExpMaxSum_x = other.wa_ExpMaxSum_x;
    wa_X_ExpMaxSum_x = other.wa_X_ExpMaxSum_x;

    wa_ExpMinSum_y = other.wa_ExpMinSum_y;
    wa_Y_ExpMinSum_y = other.wa_Y_ExpMinSum_y;
    wa_ExpMaxSum_y = other.wa_ExpMaxSum_y;
    wa_Y_ExpMaxSum_y = other.wa_Y_ExpMaxSum_y;

    return (*this);
  }

  void reset();

  float wa_ExpMinSum_x = 0;
  float wa_X_ExpMinSum_x = 0;
  float wa_ExpMaxSum_x = 0;
  float wa_X_ExpMaxSum_x = 0;

  float wa_ExpMinSum_y = 0;
  float wa_Y_ExpMinSum_y = 0;
  float wa_ExpMaxSum_y = 0;
  float wa_Y_ExpMaxSum_y = 0;
};
inline void WANetInfo::reset()
{
  wa_ExpMinSum_x = 0;
  wa_X_ExpMinSum_x = 0;
  wa_ExpMaxSum_x = 0;
  wa_X_ExpMaxSum_x = 0;
  wa_ExpMinSum_y = 0;
  wa_Y_ExpMinSum_y = 0;
  wa_ExpMaxSum_y = 0;
  wa_Y_ExpMaxSum_y = 0;
}

class WAWirelengthGradient : public WirelengthGradient
{
 public:
  WAWirelengthGradient() = delete;
  explicit WAWirelengthGradient(TopologyManager* topology_manager);
  WAWirelengthGradient(const WAWirelengthGradient&) = delete;
  WAWirelengthGradient(WAWirelengthGradient&&) = delete;
  ~WAWirelengthGradient() override = default;

  WAWirelengthGradient& operator=(const WAWirelengthGradient&) = delete;
  WAWirelengthGradient& operator=(WAWirelengthGradient&&) = delete;

  void updateWirelengthForce_OLD(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num);
  void updateWirelengthForce(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num) override;
  void updateWirelengthForceDirect(float coeff_x, float coeff_y, float min_force_bar, int32_t thread_num, GridManager* grid_manager);

  Point<float> obtainWirelengthGradient_OLD(int32_t inst_id, float coeff_x, float coeff_y);
  Point<float> obtainWirelengthGradient(int32_t inst_id, float coeff_x, float coeff_y) override;
  Point<float> obtainPinWirelengthGradient(Node* pin, float coeff_x, float coeff_y);

  // Debug
  void waWLAnalyzeForDebug(float coeff_x, float coeff_y) override;

 private:
  std::vector<WAPinInfo> _wa_pin_info_list;
  std::vector<WANetInfo> _wa_net_info_list;

  std::vector<float> _pin_grad_x_list;
  std::vector<float> _pin_grad_y_list;

  void initWAInfo();
  void resetWAPinInfo() {}
  void resetWANetInfo() {}
};

}  // namespace ipl

#endif