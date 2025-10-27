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
#ifndef IDBFASTSETUP_H
#define IDBFASTSETUP_H

#include <list>
#include <map>

#include "builder.h"
#include "dbsetup.h"
#include "def_service.h"
#include "file_cts.h"
#include "guiattribute.h"
#include "guiblock.h"
#include "guicore.h"
#include "guidie.h"
#include "guiflightline.h"
#include "guigr.h"
#include "guigraphicsscene.h"
#include "guipad.h"
#include "guipin.h"
#include "guipower.h"
#include "guirow.h"
#include "guispeedupdesign.h"
#include "guispeedupinstance.h"
#include "guispeedupitemsearch.h"
#include "guispeedupvia.h"
#include "guispeedupwire.h"
#include "guistandardcell.h"
#include "guistring.h"
#include "guivia.h"
#include "ids.hpp"
#include "lef_service.h"
#include "transform.h"
#include "vec_net.h"

using namespace idb;

namespace iplf {
  struct FileInstance;
}

#define _DEBUG_DATA_ 1
class IdbSpeedUpSetup : public DbSetup {
 public:
  IdbSpeedUpSetup(const std::vector<std::string>& lef_paths, const std::string& def_path, GuiGraphicsScene* scene);
  IdbSpeedUpSetup(IdbBuilder* idb_builder, GuiGraphicsScene* scene, DbSetupType type = DbSetupType::kChip);
  ~IdbSpeedUpSetup();

  virtual void initDB();
  virtual void createChip();
  virtual int32_t getLayerCount();
  virtual std::vector<std::string> getLayer();
  virtual void fitView(double width = 0, double height = 0);
  virtual void update(std::string node_name, std::string parent_name) { _gui_design->update(node_name, parent_name); }

  /// getter
  Transform* get_transform() { return &_transform; }

  /// searchbox
  void search(std::string search_text);

  /// drc
  void showDrc(std::map<std::string, std::map<std::string, std::vector<ids::Violation>>>& drc_db, int max_num = -1);

  void showGraph(std::map<int, ivec::VecNet> net_map);

  /// clock tree
  void showClockTree(std::vector<iplf::CtsTreeNodeMap*>& _node_list);
  GuiSpeedupClockTreeItem* buildNodeItem(iplf::CtsTreeNode* clk_node, QPointF pt, qreal unit_step);

  /// update instance for demo
  void updateInstanceInFastMode(std::vector<iplf::FileInstance>& file_inst_list);

  /// show cell master
  void showCellMasters();

  void resetIndex() { _item_index = 0; }
  bool updateInstance();
  bool updateNet();

 private:
  IdbBuilder* _db_builder = nullptr;
  IdbDesign* _design      = nullptr;
  IdbLayout* _layout      = nullptr;

  Transform _transform;
  int32_t _unit;

  GuiSpeedupDesign* _gui_design;

  int wire_number = 0;

  GuiSpeedupItemSearch* _search_item = nullptr;
  int _item_index                    = 0;

  double _fit_lx = 0;
  double _fit_ly = 0;
  double _fit_ux = 0;
  double _fit_uy = 0;
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// setter
  void createCore();
  void createDie();
  void createRow();
  /// IO
  void createIO();
  void createIOPinPortShape(vector<IdbPin*>& pin_list);
  /// Instance
  void createInstance(IdbInstanceList* inst_list = nullptr);
  void createInstanceCore(IdbInstance* instance);
  void createInstancePad(IdbInstance* instance);
  void createInstanceBlock(IdbInstance* instance);
  void createInstanceCorePin(vector<IdbPin*>& pin_list, GuiSpeedupItem* item = nullptr);
  void createInstanceCoreObs(vector<idb::IdbLayerShape*>& obs_list, GuiSpeedupItem* item = nullptr);
  void createInstanceMacroPin(vector<IdbPin*>& pin_list, GuiInstance* gui_instance);
  void createInstanceLayerShape(IdbLayerShape& layer_shape);
  ////special nets
  void createSpecialNet();
  void createSpecialNetVia(IdbSpecialWireSegment* segment, bool b_vdd = false);
  void createSpecialNetPoints(IdbSpecialWireSegment* segment, GuiSpeedupItem* item);
  /// nets
  void createNet();
  void createNetVia(IdbRegularWireSegment* segment, GuiSpeedupItemType gui_type);
  void createNetRect(IdbRegularWireSegment* segment, GuiSpeedupItem* item);
  void createNetPoints(IdbRegularWireSegment* segment, GuiSpeedupItem* item);
  GuiSpeedupItemType getNetGuiType(IdbNet* net);

  /// Tracks
  void createTrackGrid();
  void createTrackGridPreferDirection();
  void createTrackGridNonPreferDirection();

  /// Blockage
  void createBlockage();

  /// DRC
  void createDrc(GuiSpeedupDrcList* drc_list, ids::Violation& drc_db);

  void createPinPortShape(vector<IdbPin*>& pin_list, GuiSpeedupItem* item = nullptr);
  void createLayerShape(IdbLayerShape& layer_shape, GuiSpeedupItem* item = nullptr);

  /// operator
  void initTransformer();
  void initGuiDB();

  GuiSpeedupItem* findNetItem(IdbRegularWireSegment* segment, GuiSpeedupItemType gui_type);
  GuiSpeedupItem* findNetItem(GuiSpeedupWireContainer* container, IdbRegularWireSegment* segment);
  GuiSpeedupItem* findSpecialNetItem(GuiSpeedupWireContainer* container, IdbSpecialWireSegment* segment);
  GuiSpeedupItem* findViaItem(IdbLayerShape& layer_shape, GuiSpeedupItemType gui_type);

  GuiSpeedupGrid* findGridItem(GuiSeedupGridContainer* container, std::string layer_name, QPointF pt1, QPointF pt2);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// searchbox
  void initSearchbox() {
    if (_search_item == nullptr) {
      _search_item = new GuiSpeedupItemSearch();
      _search_item->setZValue(_layout->get_layers()->get_layers_num());
      addItem(_search_item);
    } else {
      _search_item->clear();
    }
  }

  void viewSearchBox() {
    _scene->viewRect(_search_item->boundingRect());
    sleep(100);
    _search_item->update();
  }

  bool highLightCoordinate(std::string name);
  bool highLightNet(std::string name);
  bool highLightNet(IdbNet* net);
  bool highLightNetPin(IdbNet* net);
  bool highLightNetWire(IdbNet* net);
  bool highLightNetFlyline(IdbNet* net);
  bool updateSearchNet(IdbNet* net);
  bool highLightInstance(std::string name);

  /// tree opration
  qreal get_leaf_step(std::vector<iplf::CtsTreeNodeMap*>& _node_list);
  qreal get_min_delay(iplf::CtsTreeNode* clk_node);
  void buildChildNode(GuiSpeedupClockTreeItem* item, iplf::CtsTreeNode* clk_node, QPointF pt, qreal unit_step);
  GuiSpeedupClockTreeItem* addPoint(GuiSpeedupClockTreeItem* item, QPointF pt, QPointF child_pt);
  int64_t get_nodelist_child_num(std::vector<iplf::CtsTreeNode*>& node_list);
};

#endif  // IDBFASTSETUP_H
