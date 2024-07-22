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
#ifndef SRC_EVALUATOR_SOURCE_CONGESTION_CONGESTIONEVAL_HPP_
#define SRC_EVALUATOR_SOURCE_CONGESTION_CONGESTIONEVAL_HPP_

#include <map>
#include <variant>

#include "CongBin.hpp"
#include "CongInst.hpp"
#include "CongNet.hpp"
#include "CongTile.hpp"
#include "idm.h"

namespace eval {
// Define a type alias for the variant inside the map
using ValueVariant = std::variant<float, std::string>;
// Define a type alias for your variants
using MacroVariant = std::map<std::string, ValueVariant>;

class CongestionEval
{
 public:
  CongestionEval()
  {
    _tile_grid = new TileGrid();
    _cong_grid = new CongGrid();
  }
  ~CongestionEval()
  {
    delete _tile_grid;
    delete _cong_grid;
  }

  void initCongGrid(const int bin_cnt_x, const int bin_cnt_y);
  void initCongInst();
  void initCongNetList();
  void mapInst2Bin();
  void mapNetCoord2Grid();

  void evalInstDens(INSTANCE_STATUS inst_status, bool eval_flip_flop = false);
  void evalPinDens(INSTANCE_STATUS inst_status, int level = 0);
  void evalNetDens(INSTANCE_STATUS inst_status);
  void evalLocalNetDens();
  void evalGlobalNetDens();

  void plotBinValue(const string& plot_path, const string& output_file_name, CONGESTION_TYPE cong_type);

  int32_t evalInstNum(INSTANCE_STATUS inst_status);
  int32_t evalNetNum(NET_CONNECT_TYPE net_type);
  int32_t evalPinTotalNum(INSTANCE_STATUS inst_status = INSTANCE_STATUS::kNone);
  int32_t evalRoutingLayerNum();
  int32_t evalTrackNum(DIRECTION direction);
  std::vector<int64_t> evalChipWidthHeightArea(CHIP_REGION_TYPE chip_region_type);
  vector<pair<string, pair<int32_t, int32_t>>> evalInstSize(INSTANCE_STATUS inst_status);
  vector<pair<string, pair<int32_t, int32_t>>> evalNetSize();

  void evalNetCong(RUDY_TYPE rudy_type, DIRECTION direction = DIRECTION::kNone);
  void plotTileValue(const string& plot_path, const string& output_file_name);

  float evalAreaUtils(INSTANCE_STATUS inst_status);
  int64_t evalArea(INSTANCE_STATUS inst_status);

  std::vector<int64_t> evalMacroPeriBias();

  int32_t evalRmTrackNum();
  int32_t evalOfTrackNum();

  int32_t evalMacroGuidance(int32_t cx, int32_t cy, int32_t width, int32_t height, const string& name);
  double evalMacroChannelUtil(float dist_ratio);
  double evalMacroChannelPinRatio(float dist_ratio);

  std::vector<MacroVariant> evalMacrosInfo();
  void plotMacroChannel(float dist_ratio, const std::string& filename);
  void evalMacroMargin();
  double evalMaxContinuousSpace();
  void evalIOPinAccess(const std::string& filename);

  // refactor
  void evalMacroDens();
  void evalMacroPinDens();
  void evalCellPinDens();
  void evalMacroChannel(float die_size_ratio = 0.5);
  void evalCellHierarchy(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroHierarchy(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroConnection(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroPinConnection(const std::string& plot_path, int level = 1, int forward = 1);
  void evalMacroIOPinConnection(const std::string& plot_path, int level = 1, int forward = 1);

  /////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////
  void reportCongestion(const std::string& plot_path, const std::string& output_file_name);

  void evalPinNum();
  void reportPinNum();
  void plotPinNum(const std::string& plot_path, const std::string& output_file_name);
  int getBinPinNum(const int& index_x, const int& index_y);
  double getBinPinDens(const int& index_x, const int& index_y);
  std::vector<float> evalPinDens();
  void evalInstDens();
  void reportInstDens();
  double getBinInstDens(const int& index_x, const int& index_y);
  std::vector<float> getInstDens();

  void evalNetCong(const std::string& rudy_type);
  void reportNetCong();
  double getBinNetCong(const int& index_x, const int& index_y, const std::string& rudy_type);
  std::vector<float> getNetCong(const std::string& rudy_type);
  // void evalViaDens();
  // void reportViaDens();
  // void plotViaDens(const std::string& plot_path, const std::string& output_file_name);
  // double getBinViaDens(const int& index_x, const int& index_y);
  // std::vector<CongNet>& getPostRouteNetlist();

  std::vector<float> evalRouteCong();
  float evalACE(const std::vector<float>& hor_edge_cong_list, const std::vector<float>& ver_edge_cong_list);
  std::vector<int> evalOverflow();                 // <TOF,MOF>
  std::vector<int> evalOverflow(int layer_index);  // <TOF,MOF>
  int32_t evalRemain(int layer_index);
  std::vector<float> getUseCapRatioList();
  // produce layer_num maps: raw overflow
  void plotGRCong(const std::string& plot_path, const std::string& output_file_name);
  void plotGRCongOneLayer(const std::string& plot_path, const std::string& output_file_name, int layer_index);
  // produce two maps: TOF,MOF
  void plotOverflow(const std::string& plot_path, const std::string& output_file_name);
  void plotOverflow(const std::string& plot_path, const std::string& output_file_name, const std::vector<int>& plane_grid, const int& x_cnt,
                    const std::string& type);
  void reportCongMap();

  /*----Common used----*/
  void set_tile_grid(TileGrid* tile_grid) { _tile_grid = tile_grid; }
  void set_tile_grid(const int& lx, const int& ly, const int& tileCntX, const int& tileCntY, const int& tileSizeX, const int& tileSizeY,
                     const int& numRoutingLayers);
  void set_cong_grid(CongGrid* cong_grid) { _cong_grid = cong_grid; }
  void set_cong_grid(const int& lx, const int& ly, const int& binCntX, const int& binCntY, const int& binSizeX, const int& binSizeY);
  void set_cong_inst_list(const std::vector<CongInst*>& cong_inst_list) { _cong_inst_list = cong_inst_list; }
  void set_cong_net_list(const std::vector<CongNet*>& cong_net_list) { _cong_net_list = cong_net_list; }

  CongGrid* get_cong_grid() const { return _cong_grid; }
  TileGrid* get_tile_grid() const { return _tile_grid; }
  std::vector<CongInst*>& get_cong_inst_list() { return _cong_inst_list; }
  std::vector<CongNet*>& get_cong_net_list() { return _cong_net_list; }

  void checkRUDYType(const std::string& rudy_type);
  void reportTileGrid();
  void reportCongGrid();

 private:
  TileGrid* _tile_grid = nullptr;
  CongGrid* _cong_grid = nullptr;
  std::vector<CongInst*> _cong_inst_list;
  std::vector<CongNet*> _cong_net_list;
  std::map<std::string, CongInst*> _name_to_inst_map;

  int32_t getOverlapArea(CongBin* bin, CongInst* inst);
  int32_t getOverlapArea(CongBin* bin, CongNet* net);

  double getRudy(CongBin* bin, CongNet* net, DIRECTION direction = DIRECTION::kNone);
  double getRudyDev(CongBin* bin, CongNet* net);
  double getPinRudy(CongBin* bin, CongNet* net, DIRECTION direction = DIRECTION::kNone);
  double getTrueRudy(CongBin* bin, CongNet* net, const std::map<std::string, int64_t>& map);
  float calcLness(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax);
  int64_t calcLowerLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymin);
  int64_t calcLowerRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymin);
  int64_t calcUpperLeftRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmin, int32_t ymax);
  int64_t calcUpperRightRP(std::vector<std::pair<int32_t, int32_t>>& point_set, int32_t xmax, int32_t ymax);
  double getLUT(const int32_t& pin_num, const int32_t& aspect_ratio, const float& l_ness);

  float getUsageCapacityRatio(Tile* tile);
  CongPin* wrapCongPin(idb::IdbPin* idb_pin);
  std::string fixSlash(std::string raw_str);
  std::vector<std::string> splitHier(const std::string& str, char delimiter);
  std::string getHierByLevel(const std::string& hier, int level, int forward);
  std::string getPrefix(const std::string& pin_name);

  std::string instanceStatusToString(INSTANCE_STATUS inst_status);
};

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_CONGESTION_CONGESTIONEVAL_HPP_
