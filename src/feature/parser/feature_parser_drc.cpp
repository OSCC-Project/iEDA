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
 * @project		iEDA
 * @file		feature_parser.cpp
 * @author		Yell
 * @date		10/08/2023
 * @version		0.1
 * @description


        feature parser
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbDesign.h"
#include "IdbInstance.h"
#include "feature_drc.h"
#include "feature_parser.h"
#include "json_parser.h"

namespace ieda_feature {

bool FeatureParser::buildMacroDrc(std::string path, std::string drc_path)
{
  std::cout << "path : " << path << std::endl;
  std::cout << "drc_path : " << drc_path << std::endl;

  DrcMacroDistribution macro_drc;
  /// init macro info
  auto macro_list = _design->get_instance_list()->get_macro_list();
  for (auto macro : macro_list) {
    DrcMacroCount drc_macro{};
    drc_macro.name = macro->get_name();
    drc_macro.llx = macro->get_bounding_box()->get_low_x();
    drc_macro.lly = macro->get_bounding_box()->get_low_y();
    drc_macro.urx = macro->get_bounding_box()->get_high_x();
    drc_macro.ury = macro->get_bounding_box()->get_high_y();
    drc_macro.drc_num = 0;
    macro_drc.macro_list[macro->get_name()] = drc_macro;
  }

  std::cout << "macro size : " << macro_drc.macro_list.size() << std::endl;

  /// load drc json
  auto json_file = std::ifstream(drc_path);
  if (false == json_file.is_open()) {
    std::cout << "Error, can't open file : " << drc_path << std::endl;
    return false;
  }

  json root;
  json_file >> root;

  auto json_distribution = root["drc"]["distribution"];
  auto drc_total_number = root["drc"]["number"];

  for (json::iterator item_drc = json_distribution.begin(); item_drc != json_distribution.end(); ++item_drc) {
    auto json_drc = item_drc.value();
    auto json_layers = json_drc["layers"];

    for (json::iterator item_layer = json_layers.begin(); item_layer != json_layers.end(); ++item_layer) {
      auto json_layer = item_layer.value();
      auto json_list = json_layer["list"];

      for (json::iterator item_rect = json_list.begin(); item_rect != json_list.end(); ++item_rect) {
        auto json_rect = item_rect.value();

        DrcRect rect;
        rect.llx = _design->transUnitDB(json_rect["llx"]);
        rect.lly = _design->transUnitDB(json_rect["lly"]);
        rect.urx = _design->transUnitDB(json_rect["urx"]);
        rect.ury = _design->transUnitDB(json_rect["ury"]);

        /// process drc rect
        for (auto& [name, macro] : macro_drc.macro_list) {
          /// check if rect intersected with macro
          if (rect.llx > macro.urx || rect.urx < macro.llx || rect.lly > macro.ury || rect.ury < macro.lly) {
            /// not intersect
            continue;
          }

          macro.drc_num += 1;
          break;
        }
      }
    }
  }

  /// save macro drc distribution to json
  std::ofstream& file_stream = ieda::getOutputFileStream(path);
  json root_output;

  int macro_drc_num = 0;
  int macro_num = 0;
  root_output["drc_total_number"] = drc_total_number;
  root_output["macro_drc_num"] = macro_drc_num;
  root_output["macro_num"] = macro_num;
  root_output["macro_list"] = {};
  for (auto& [name, macro] : macro_drc.macro_list) {
    if (macro.drc_num <= 0) {
      continue;
    }

    json json_macro;
    json_macro["llx"] = macro.llx;
    json_macro["lly"] = macro.lly;
    json_macro["urx"] = macro.urx;
    json_macro["ury"] = macro.ury;
    json_macro["drc_num"] = macro.drc_num;

    root_output["macro_list"][name] = json_macro;

    std::cout << "macro " << name << " " << macro.drc_num << " " << macro.llx << " " << macro.lly << " " << macro.urx << " " << macro.ury
              << std::endl;
    macro_num++;
    macro_drc_num += macro.drc_num;
  }

  root_output["macro_drc_num"] = macro_drc_num;
  root_output["macro_num"] = macro_num;

  /// build route data json
  file_stream << std::setw(4) << root_output;
  ieda::closeFileStream(file_stream);

  return true;
}

}  // namespace ieda_feature
