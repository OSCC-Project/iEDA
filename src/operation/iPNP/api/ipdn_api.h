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
#pragma once

#include <any>
#include <map>
#include <string>
#include <vector>

#define pdnApiInst ipdn::PdnApi::getInstance()
namespace ipdn {
class PdnApi
{
 public:
  static PdnApi* getInstance()
  {
    if (!_instance) {
      _instance = new PdnApi;
    }
    return _instance;
  }

  static void destroyInst()
  {
    if (_instance != nullptr) {
      delete _instance;
      _instance = nullptr;
    }
  }

  // function
  void addIOPin(std::string pin_name, std::string net_name, std::string direction, bool is_power = true);
  void globalConnect(const std::string pdn_net_name, const std::string instance_pdn_pin_name, bool is_power);
  void placePdnPort(std::string pin_name, std::string io_cell_name, int32_t offset_x, int32_t offset_y, int32_t width, int32_t height,
                    std::string layer_name);
  void createGrid(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width, double route_offset);
  void createStripe(std::string power_net_name, std::string ground_net_name, std::string layer_name, double route_width, double pitchh,
                    double route_offset);
  void connectLayerList(std::vector<std::string>& layer_list);

  void connectMacroToPdnGrid(std::vector<std::string> power_name, std::vector<std::string> ground_name, std::string layer_name_first,
                             std::string layer_name_second, std::string orient);

  void connectIOPinToPowerStripe(std::vector<double>& point_list, const std::string layer_name);
  void connectPowerStripe(std::vector<double>& point_list, const std::string& net_name, const std::string& layer_name, int32_t width);

  bool addSegmentStripeList(std::vector<double>& point_list, std::string net_name, std::string layer_name, int32_t width);
  bool addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, double coord_x, double coord_y, int32_t width,
                     int32_t height);
  bool addSegmentVia(std::string net_name, std::string top_metal, std::string bottom_metal, int32_t coord_x, int32_t coord_y, int32_t width,
                     int32_t height);
  bool addSegmentVia(std::string net_name, std::string cut_layer_name, int32_t coord_x, int32_t coord_y, int32_t width, int32_t height);

 private:
  static PdnApi* _instance;

  PdnApi() {}
  PdnApi(const PdnApi& other) = delete;
  PdnApi(PdnApi&& other) = delete;
  ~PdnApi() = default;
  PdnApi& operator=(const PdnApi& other) = delete;
  PdnApi& operator=(PdnApi&& other) = delete;

  // function
};

}  // namespace ipdn
