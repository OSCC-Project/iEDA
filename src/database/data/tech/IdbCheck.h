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
