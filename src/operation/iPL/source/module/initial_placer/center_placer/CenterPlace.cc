/*
 * @Author: S.J Chen
 * @Date: 2022-04-01 12:07:29
 * @LastEditTime: 2022-04-01 12:50:51
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/pre_placer/center_place/CenterPlace.cc
 * Contact : https://github.com/sjchanson
 */

#include "CenterPlace.hh"

namespace ipl {

void CenterPlace::runCenterPlace()
{
  Point<int32_t> core_center = _placer_db->get_layout()->get_core_shape().get_center();

  auto inst_list = _placer_db->get_design()->get_instance_list();
  for (auto* inst : inst_list) {
    if (inst->isUnPlaced()) {
      inst->update_center_coordi(core_center);
    }
  }
}

}  // namespace ipl