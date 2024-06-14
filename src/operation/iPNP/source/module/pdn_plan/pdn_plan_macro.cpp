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
#include "idm.h"
#include "pdn_plan.h"
#include "pdn_via.h"

namespace ipdn {

/**
 * @brief 将IdbOrient转换为string类型，方便后续判断
 *
 * @param ori
 * @return pair<std::string, std::string>
 */
pair<std::string, std::string> PdnPlan::orientToStr(IdbOrient ori)
{
  pair<std::string, std::string> result;
  if (ori == IdbOrient::kN_R0) {
    result.first = "N";
    result.second = "R0";
  } else if (ori == IdbOrient::kE_R270) {
    result.first = "E";
    result.second = "R270";
  } else if (ori == IdbOrient::kS_R180) {
    result.first = "S";
    result.second = "R180";
  } else if (ori == IdbOrient::kW_R90) {
    result.first = "W";
    result.second = "R90";
  } else if (ori == IdbOrient::kFS_MX) {
    result.first = "FS";
    result.second = "MX";
  } else if (ori == IdbOrient::kFN_MY) {
    result.first = "FN";
    result.second = "MY";
  } else if (ori == IdbOrient::kFW_MX90) {
    result.first = "FW";
    result.second = "MX90";
  } else if (ori == IdbOrient::kFE_MY90) {
    result.first = "FE";
    result.second = "ME90";
  }
  return result;
}

/**
 * @brief
 * 批量进行macro连接电源的操作，主要是循环调用针对单个macro进行电源连接的接口
 *
 * @param power_name
 * @param ground_name
 * @param pin_layer
 * @param pdn_layer
 * @param orientt
 */
void PdnPlan::connectMacroToPdnGrid(std::vector<std::string> power_name, std::vector<std::string> ground_name, std::string pin_layer,
                                    std::string pdn_layer, std::string orientt)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_inst_list = idb_design->get_instance_list();

  for (IdbInstance* macro : idb_inst_list->get_instance_list()) {
    if (macro->is_fixed() || macro->is_placed()) {
      IdbOrient orient = macro->get_orient();
      pair<std::string, std::string> ori = orientToStr(orient);
      if (orientt.find(ori.first) || orientt.find(ori.second)) {
        connectMacroToPdnGrid(macro, power_name, ground_name, pin_layer, pdn_layer);
      }
    }
  }
}

/**
 * @brief Used to realize the connection between macro and power network
 *
 * @param macro
 * @param power_name
 * @param ground_name
 * @param pin_layer
 * @param pdn_layer
 */
void PdnPlan::connectMacroToPdnGrid(idb::IdbInstance* macro, std::vector<std::string> power_name, std::vector<std::string> ground_name,
                                    std::string pin_layer, std::string pdn_layer)
{
  auto idb_design = dmInst->get_idb_design();
  auto idb_layout = dmInst->get_idb_layout();
  auto idb_pdn_list = idb_design->get_special_net_list();
  auto idb_layer_list = idb_layout->get_layers();
  // auto via_list = idb_design->get_via_list();

  // map<power_pin_name,map<layer_name,rect_list>>PdnPlan
  std::map<std::string, std::map<std::string, std::vector<IdbRect>>> power, ground;
  initMacroPowerPinShape(macro, power_name, ground_name, power, ground);

  idb::IdbSpecialNet* vdd = idb_pdn_list->find_net("VDD");
  idb::IdbSpecialNet* vss = idb_pdn_list->find_net("VSS");

  idb::IdbSpecialWire* vdd_wire = vdd->get_wire_list()->get_wire_list()[0];
  idb::IdbSpecialWire* vss_wire = vss->get_wire_list()->get_wire_list()[0];

  /// 现在默认macro的电源相关引脚都在metal4，需要之后修改
  /// hard code
  std::vector<idb::IdbRect> macro_vdd_port = power["VDD"][pin_layer];
  std::vector<idb::IdbRect> macro_vss_port = ground["VSS"][pin_layer];

  std::vector<idb::IdbRect> via_shape_vdd;
  std::vector<idb::IdbRect> via_shape_vss;
  ///
  via_shape_vdd = findOverlapbetweenMacroPdnAndStripe(macro_vdd_port, pdn_layer, vdd_wire);
  via_shape_vss = findOverlapbetweenMacroPdnAndStripe(macro_vss_port, pdn_layer, vss_wire);

  idb::IdbLayerRouting* layer_bottom = dynamic_cast<idb::IdbLayerRouting*>(idb_layer_list->find_layer(pin_layer));
  idb::IdbLayerRouting* layer_top = dynamic_cast<idb::IdbLayerRouting*>(idb_layer_list->find_layer(pdn_layer));
  if (layer_top->get_order() < layer_bottom->get_order()) {
    std::swap(layer_top, layer_bottom);
  }

  vector<IdbVia*> via_find_list;
  PdnVia pdn_via;
  // VDD
  for (idb::IdbRect rect : via_shape_vdd) {
    for (int32_t layer_order = layer_bottom->get_order(); layer_order <= (layer_top->get_order() - 2);) {
      // find layer cut
      idb::IdbLayerCut* layer_cut_find = dynamic_cast<idb::IdbLayerCut*>(idb_layer_list->find_layer_by_order(layer_order + 1));
      if (layer_cut_find == nullptr) {
        std::cout << "Error : layer input illegal." << std::endl;
        return;
      }
      //   IdbVia *via_find = via_list->find_via_generate(
      //       layer_cut_find, rect.get_width(), rect.get_height());

      idb::IdbVia* via_find = pdn_via.findVia(layer_cut_find, rect.get_width(), rect.get_height());
      if (via_find == nullptr) {
        std::cout << "Error : can not find VIA matchs." << std::endl;
        continue;
      }
      idb::IdbLayer* layer_topp = via_find->get_top_layer_shape().get_layer();
      idb::IdbCoordinate<int32_t> coordinate = rect.get_middle_point();
      idb::IdbSpecialWireSegment* segment_via
          = pdn_via.createSpecialWireVia(layer_topp, 0, idb::IdbWireShapeType::kStripe, &coordinate, via_find);
      vdd_wire->add_segment(segment_via);
      layer_order += 2;
    }
  }

  // VSS
  for (idb::IdbRect rect : via_shape_vss) {
    for (int32_t layer_order = layer_bottom->get_order(); layer_order <= (layer_top->get_order() - 2);) {
      // find layer cut
      idb::IdbLayerCut* layer_cut_find = dynamic_cast<idb::IdbLayerCut*>(idb_layer_list->find_layer_by_order(layer_order + 1));
      if (layer_cut_find == nullptr) {
        std::cout << "Error : layer input illegal." << std::endl;
        return;
      }
      //   IdbVia *via_find = via_list->find_via_generate(
      //       layer_cut_find, rect.get_width(), rect.get_height());
      idb::IdbVia* via_find = pdn_via.findVia(layer_cut_find, rect.get_width(), rect.get_height());
      if (via_find == nullptr) {
        std::cout << "Error : can not find VIA matchs." << std::endl;
        continue;
      }
      idb::IdbCoordinate<int32_t> coordinate = rect.get_middle_point();
      idb::IdbSpecialWireSegment* segment_via
          = pdn_via.createSpecialWireVia(layer_top, 0, idb::IdbWireShapeType::kStripe, &coordinate, via_find);
      vss_wire->add_segment(segment_via);
      layer_order += 2;
    }
  }
}

/**
 * @brief
 * 110工艺中的macro，引脚的port由很多个矩形块构成；规律是每两个相邻的矩形块会构成一个大的矩形块。
 * 该函数会将有重叠或是紧密相连的金属块进行合并，生成新的大矩形块
 *
 * @param macro
 * @param power_name
 * @param ground_name
 * @param power
 * @param ground
 */
void PdnPlan::initMacroPowerPinShape(idb::IdbInstance* macro, std::vector<std::string> power_name, std::vector<std::string> ground_name,
                                     std::map<std::string, std::map<std::string, std::vector<idb::IdbRect>>>& power,
                                     std::map<std::string, std::map<std::string, std::vector<idb::IdbRect>>>& ground)
{
  std::vector<idb::IdbPin*> power_pins;
  std::vector<idb::IdbPin*> ground_pins;

  // get power and ground pins
  for (auto pn : power_name) {
    idb::IdbPin* ppin = macro->get_pin_list()->find_pin(pn);
    if (ppin == nullptr) {
      std::cout << "No Power Pin: " << pn << std::endl;
      delete ppin;
    } else {
      power_pins.push_back(ppin);
    }
  }
  for (auto pn : ground_name) {
    idb::IdbPin* gpin = macro->get_pin_list()->find_pin(pn);
    if (gpin == nullptr) {
      std::cout << "No Ground Pin: " << pn << std::endl;
      delete gpin;
    } else {
      ground_pins.push_back(gpin);
    }
  }

  // get rect list of port, default all port layershape are on METAL4
  for (idb::IdbPin* pin : power_pins) {
    std::map<std::string, std::vector<idb::IdbRect>> port_list = mergeOverlapRect(pin);
    power.insert(make_pair(pin->get_pin_name(), port_list));
  }
  for (idb::IdbPin* pin : ground_pins) {
    std::map<std::string, std::vector<idb::IdbRect>> port_list = mergeOverlapRect(pin);
    ground.insert(make_pair(pin->get_pin_name(), port_list));
  }
  std::cout << "Init macro power port shape" << std::endl;
}

/**
 * @brief 寻找macro的电源引脚与电源网络相交叉的矩形块
 *
 * @param rect_list
 * @param layer
 * @param sp_wire
 * @return std::vector<IdbRect>
 */
std::vector<idb::IdbRect> PdnPlan::findOverlapbetweenMacroPdnAndStripe(std::vector<idb::IdbRect> rect_list, const std::string& layer,
                                                                       idb::IdbSpecialWire* sp_wire)
{
  std::vector<idb::IdbRect> result;
  for (idb::IdbRect rect : rect_list) {
    for (idb::IdbSpecialWireSegment* seg : sp_wire->get_segment_list()) {
      if (!rect.isIntersection(*(seg->get_bounding_box()))) {
        continue;
      } else if (seg->get_layer()->get_name() == layer) {
        int32_t rect_llx = rect.get_low_x();
        int32_t rect_lly = rect.get_low_y();
        int32_t rect_urx = rect.get_high_x();
        int32_t rect_ury = rect.get_high_y();
        int32_t seg_llx = seg->get_bounding_box()->get_low_x();
        int32_t seg_lly = seg->get_bounding_box()->get_low_y();
        int32_t seg_urx = seg->get_bounding_box()->get_high_x();
        int32_t seg_ury = seg->get_bounding_box()->get_high_y();
        std::vector<int32_t> x;
        x.push_back(rect_llx);
        x.push_back(rect_urx);
        x.push_back(seg_llx);
        x.push_back(seg_urx);
        std::vector<int32_t> y;
        y.push_back(rect_lly);
        y.push_back(rect_ury);
        y.push_back(seg_lly);
        y.push_back(seg_ury);
        sort(x.begin(), x.end(), [](int32_t a, int32_t b) { return a < b; });
        sort(y.begin(), y.end(), [](int32_t a, int32_t b) { return a < b; });
        IdbRect overlap = IdbRect(x[1], y[1], x[2], y[2]);
        result.push_back(overlap);
      }
    }
  }
  return result;
}

/**
 * @brief 合并有重叠的矩形
 *
 * @param pin
 * @return std::map<std::string, std::vector<IdbRect>>
 */
std::map<std::string, std::vector<idb::IdbRect>> PdnPlan::mergeOverlapRect(idb::IdbPin* pin)
{
  std::map<std::string, std::vector<idb::IdbRect*>> temp;
  std::map<std::string, std::vector<idb::IdbRect>> result;
  std::vector<idb::IdbLayerShape*> box_list = pin->get_port_box_list();
  std::vector<idb::IdbRect*> rect_list;
  // get rect list of port, default all port layershape are on METAL4
  for (idb::IdbLayerShape* shape : box_list) {
    std::string layer_name = shape->get_layer()->get_name();
    // std::vector<IdbRect> rect_list =
    // mergeOverlapRect(shape->get_rect_list());
    if (temp.count(layer_name) == 0) {
      temp.insert(make_pair(layer_name, shape->get_rect_list()));
    } else {
      for (idb::IdbRect* rect : shape->get_rect_list()) {
        temp[layer_name].push_back(rect);
      }
    }
  }
  for (auto a : temp) {
    result.insert(make_pair(a.first, mergeOverlapRect(a.second)));
  }
  return result;
}

/**
 * @brief 批量进行合并矩形的操作
 *
 * @param rect_list
 * @return std::vector<IdbRect>
 */
std::vector<idb::IdbRect> PdnPlan::mergeOverlapRect(std::vector<idb::IdbRect*> rect_list)
{
  std::vector<idb::IdbRect> result;
  if (rect_list.size() == 0) {
    return result;
  }

  for (size_t i = 0; i < rect_list.size() - 1; ++i) {
    if ((rect_list[i]->get_high_x() >= rect_list[i + 1]->get_low_x()) && (rect_list[i]->get_high_x() <= rect_list[i + 1]->get_high_x())
        && (rect_list[i]->get_low_y() == rect_list[i + 1]->get_low_y())) {
      idb::IdbRect rect = idb::IdbRect(rect_list[i]->get_low_x(), rect_list[i]->get_low_y(), rect_list[i + 1]->get_high_x(),
                                       rect_list[i + 1]->get_high_y());
      result.push_back(rect);
      ++i;
    } else if ((rect_list[i]->get_high_y() >= rect_list[i + 1]->get_low_y())
               && (rect_list[i]->get_high_y() <= rect_list[i + 1]->get_high_y())
               && (rect_list[i]->get_low_x() == rect_list[i + 1]->get_low_x())) {
      idb::IdbRect rect = idb::IdbRect(rect_list[i]->get_low_x(), rect_list[i]->get_low_y(), rect_list[i + 1]->get_high_x(),
                                       rect_list[i + 1]->get_high_y());
      result.push_back(rect);
      ++i;
    } else {
      result.push_back(rect_list[i]);
    }
  }
  return result;
}

}  // namespace ipdn