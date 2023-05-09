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
#ifndef IDBSETUP_H
#define IDBSETUP_H

#include <list>
#include <map>

#include "builder.h"
#include "dbsetup.h"
#include "def_service.h"
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
#include "guistandardcell.h"
#include "guivia.h"
#include "lef_service.h"
#include "transform.h"

using namespace idb;

class IdbSetup : public DbSetup {
 public:
  //   IdbSetup(const std::map<std::string, std::list<std::string>>& filemap,
  //            GuiGraphicsScene*                                    scene);
  IdbSetup(const std::vector<std::string>& lef_paths, const std::string& def_path, GuiGraphicsScene* scene);
  IdbSetup(IdbBuilder* idb_builder, GuiGraphicsScene* scene);
  ~IdbSetup();

  virtual void initDB();
  virtual void createChip();
  virtual int32_t getLayerCount();
  virtual std::vector<std::string> getLayer();
  virtual void fitView(double width = 0, double height = 0);
  virtual void update(std::string node_name, std::string parent_name) { }

  Transform* get_transform() { return &_transform; }

  void createDbu();
  void createCore();
  void createDie();
  void createRow();
  void createIO();
  void createIOPinPortShape(vector<IdbPin*>& pin_list);
  void createInstance();
  void createInstanceCore(IdbInstance* instance);
  void createInstancePad(IdbInstance* instance);
  void createInstanceBlock(IdbInstance* instance);
  void createInstancePin(vector<IdbPin*>& pin_list, GuiInstance* gui_instance);
  void createPinPortShape(vector<IdbPin*>& pin_list);
  void createSpecialNet();
  void createSpecialNetVia(IdbSpecialWireSegment* segment);
  void createSpecialNetPoints(IdbSpecialWireSegment* segment);
  void createNet();
  void createNetVia(IdbRegularWireSegment* segment);
  void createNetRect(IdbRegularWireSegment* segment);
  void createNetPoints(IdbRegularWireSegment* segment);

  void createLayerShape(IdbLayerShape& layer_shape);

  // GuiItem::Orientation orientationTransform(IdbOrient orient_type);

  /// Test
  struct GrInfo {
    int32_t x;
    int32_t y;
    int32_t ur_x;
    int32_t ur_y;
    std::string layer_name;
    // std::string info;

    double _h_supply;
    double _v_supply;
    double _h_demand;
    double _v_demand;
    double _via_demand;
  };
  void TestGrGui();
  void createGrGui(vector<GrInfo>& info_list);
  void createGrCongestionMap(vector<GrInfo>& info_list);
  void analysisResource(const std::string& filename, vector<GrInfo>& info_list);

 private:
  IdbBuilder* _db_builder = nullptr;
  IdbDesign* _design      = nullptr;
  IdbLayout* _layout      = nullptr;

  Transform _transform;

  int32_t _unit;
};

#endif  // IDBSETUP_H
