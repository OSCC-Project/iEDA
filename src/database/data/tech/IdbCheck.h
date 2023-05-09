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
#ifndef _IDB_CHECK_H
#define _IDB_CHECK_H

#include "IdbLayout.h"
#include "IdbTech.h"

namespace idb {
class IdbCheck
{
 public:
  IdbCheck();
  ~IdbCheck();

  IdbTech* get_tech();

  int initTech(IdbLayout* layout);
  // init LayerCheckList Data include routlayerList and cutLayerList
  int initTechLayerList(IdbLayout* layout);
  int addTechRoutingLayer(IdbLayerRouting* layerRouting);
  int initTechRoutingLayer(IdbLayerRouting* layerRouting, IdbTechRoutingLayer* techRoutingLayer);
  int initDensityCheck(IdbLayerRouting* layerRouting, IdbTechRoutingLayer* techRoutingLayer);
  int initMinimumCutCheck(IdbLayerRouting* layerRouting, IdbTechRoutingLayer* techRoutingLayer);
  int initSpacingCheck(IdbLayerRouting* layerRouting, IdbTechRoutingLayer* techRoutingLayer);
  int initMinEnclosedAreaCheck(IdbLayerRouting* layerRouting, IdbTechRoutingLayer* techRoutingLayer);
  int addTechCutLayer(IdbLayerCut* layerCut);
  int initTechCutLayer(IdbLayerCut* layerCut, IdbTechCutLayer* techCutLayer);
  int initArraySpacingCheck(IdbLayerCut* layerCut, IdbTechCutLayer* techCutLayer);
  int initEnclosureCheck(IdbLayerCut* layerCut, IdbTechCutLayer* techCutLayer);
  // init viaList
  int initTechViaList(IdbLayout* layout);
  int initViaMetalRect(IdbRect* rect, IdbTechRect* techRect);
  int initViaCutRectList(IdbViaMasterFixed* viaMasterFixed, IdbTechVia* techVia);
  // init viaRuleList
  int initTechViaRuleList(IdbLayout* layout);
  int initViaRuleLayer(IdbViaRuleGenerate* viaRuleGenerate, IdbTechViaRule* techViaRule);
  int initViaRuleEnclosure(IdbViaRuleGenerate* viaRuleGenerate, IdbTechViaRule* techViaRule);
  int initViaRuleCutRect(IdbViaRuleGenerate* viaRuleGenerate, IdbTechViaRule* techViaRule);

 private:
  std::unique_ptr<IdbTech> _tech;
};
}  // namespace idb

#endif
