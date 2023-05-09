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
#include "IdbCheck.h"

#include <vector>

namespace idb {
#define kDbFail    0
#define kDbSuccess 1
  /**
   * @brief Construct a new Idb Check:: Idb Check object
   *
   */
  IdbCheck::IdbCheck() { _tech = std::make_unique<IdbTech>(); }
  IdbCheck::~IdbCheck() { }

  IdbTech *IdbCheck::get_tech() {
    if (!_tech) {
      _tech = std::make_unique<IdbTech>();
    }
    return _tech.get();
  }

  int IdbCheck::initTech(IdbLayout *layout) {
    if (layout == nullptr) {
      return kDbFail;
    }
    initTechLayerList(layout);
    initTechViaList(layout);
    initTechViaRuleList(layout);
    return kDbSuccess;
  }

  int IdbCheck::initTechLayerList(IdbLayout *layout) {
    IdbLayers *layers                 = layout->get_layers();
    std::vector<IdbLayer *> layerList = layers->get_layers();

    for (auto &layer : layerList) {
      if (IdbLayerType::kLayerCut == layer->get_type()) {
        IdbLayerCut *layerCut = dynamic_cast<IdbLayerCut *>(layer);
        addTechCutLayer(layerCut);
      } else if (IdbLayerType::kLayerRouting == layer->get_type()) {
        IdbLayerRouting *layerRouting = dynamic_cast<IdbLayerRouting *>(layer);
        addTechRoutingLayer(layerRouting);
      }
    }
    _tech->initLayerId();
    return kDbSuccess;
  }

  int IdbCheck::addTechRoutingLayer(IdbLayerRouting *layerRouting) {
    std::unique_ptr<IdbTechRoutingLayer> idbTechRoutingLayer = std::make_unique<IdbTechRoutingLayer>();
    IdbTechRoutingLayer *techRoutingLayer                    = idbTechRoutingLayer.get();

    initTechRoutingLayer(layerRouting, techRoutingLayer);

    _tech->addTechRoutingLayer(idbTechRoutingLayer);
    return kDbSuccess;
  }

  int IdbCheck::initTechRoutingLayer(IdbLayerRouting *layerRouting, IdbTechRoutingLayer *techRoutingLayer) {
    techRoutingLayer->set_name(layerRouting->get_name());
    techRoutingLayer->set_type(LayerTypeEnum::kROUTING);
    techRoutingLayer->set_direction(layerRouting->get_direction());
    techRoutingLayer->set_width(layerRouting->get_width());
    IdbLayerOrientValue pitch = layerRouting->get_pitch();  // pitch is kBothXY type
    techRoutingLayer->set_pitch(pitch.orient_x);
    IdbLayerOrientValue offset = layerRouting->get_offset();  // offset is kBothXY type
    techRoutingLayer->set_offset(offset.orient_x);
    techRoutingLayer->set_thickness(layerRouting->get_thickness());
    techRoutingLayer->set_resistance(layerRouting->get_resistance());
    techRoutingLayer->set_capacitance(layerRouting->get_capacitance());
    techRoutingLayer->set_edge_capcitance(layerRouting->get_edge_capacitance());
    std::unique_ptr<IdbMaxWidthCheck> idbMaxWidthCheck = std::make_unique<IdbMaxWidthCheck>(layerRouting->get_max_width());
    techRoutingLayer->set_max_width_check(idbMaxWidthCheck);
    std::unique_ptr<IdbMinWidthCheck> idbMinWidthCheck = std::make_unique<IdbMinWidthCheck>(layerRouting->get_min_width());
    techRoutingLayer->set_min_width_check(idbMinWidthCheck);
    std::unique_ptr<IdbMinAreaCheck> idbMinAreaCheck = std::make_unique<IdbMinAreaCheck>(layerRouting->get_area());
    techRoutingLayer->set_min_area_check(idbMinAreaCheck);

    initDensityCheck(layerRouting, techRoutingLayer);
    initSpacingCheck(layerRouting, techRoutingLayer);
    initMinEnclosedAreaCheck(layerRouting, techRoutingLayer);
    return kDbSuccess;
  }

  int IdbCheck::initMinEnclosedAreaCheck(IdbLayerRouting *layerRouting, IdbTechRoutingLayer *techRoutingLayer) {
    std::vector<IdbMinEncloseArea> encloseAreaList = layerRouting->get_min_enclose_area_list()->get_min_area_list();

    if (encloseAreaList.size() > 0) {
      int minEnclosedArea = (encloseAreaList.begin())->_area;
      std::unique_ptr<IdbMinEnclosedAreaCheck> idbMinEnclosedAreaCheck =
          std::make_unique<IdbMinEnclosedAreaCheck>(minEnclosedArea);
      techRoutingLayer->set_min_enclosed_area_check(idbMinEnclosedAreaCheck);
    }
    return kDbSuccess;
  }

  int IdbCheck::initDensityCheck(IdbLayerRouting *layerRouting, IdbTechRoutingLayer *techRoutingLayer) {
    std::unique_ptr<IdbDensityCheck> idbDensityCheck = std::make_unique<IdbDensityCheck>();
    idbDensityCheck->set_max_density(layerRouting->get_max_density());
    idbDensityCheck->set_min_density(layerRouting->get_min_density());
    idbDensityCheck->set_density_check_width(layerRouting->get_density_check_width());
    idbDensityCheck->set_density_check_length(layerRouting->get_density_check_length());
    idbDensityCheck->set_density_check_step(layerRouting->get_density_check_step());

    techRoutingLayer->set_density_check(idbDensityCheck);
    return kDbSuccess;
  }

  int IdbCheck::initMinimumCutCheck(IdbLayerRouting *layerRouting, IdbTechRoutingLayer *techRoutingLayer) {
    // std::unique_ptr<IdbMinimumCutCheck> idbDensityCheck = std::make_unique<IdbMinimumCutCheck>();
    return kDbSuccess;
  }

  int IdbCheck::initSpacingCheck(IdbLayerRouting *layerRouting, IdbTechRoutingLayer *techRoutingLayer) {
    IdbLayerSpacingList *layerSpacingList      = layerRouting->get_spacing_list();
    std::vector<IdbLayerSpacing *> spacingList = layerSpacingList->get_spacing_list();
    for (auto spacing : spacingList) {
      if (spacing->get_spacing_type() == IdbLayerSpacingType::kSpacingDefault) {
        std::unique_ptr<IdbSpacingCheck> idbSpacingCheck = std::make_unique<IdbSpacingCheck>(spacing->get_min_spacing());
        techRoutingLayer->set_spacing_check(idbSpacingCheck);
      } else if (spacing->get_spacing_type() == IdbLayerSpacingType::kSpacingRange) {
        std::unique_ptr<IdbSpacingRangeCheck> idbSpacingRangeCheck = std::make_unique<IdbSpacingRangeCheck>();
        idbSpacingRangeCheck->set_min_width(spacing->get_min_width());
        idbSpacingRangeCheck->set_max_width(spacing->get_max_width());
        idbSpacingRangeCheck->set_min_spacing(spacing->get_min_spacing());
        techRoutingLayer->addSpacingRangeCheck(idbSpacingRangeCheck);
      }
    }
    return kDbSuccess;
  }

  int IdbCheck::addTechCutLayer(IdbLayerCut *layerCut) {
    std::unique_ptr<IdbTechCutLayer> idbTechCutLayer = std::make_unique<IdbTechCutLayer>();
    IdbTechCutLayer *techCutLayer                    = idbTechCutLayer.get();

    initTechCutLayer(layerCut, techCutLayer);

    _tech->addTechCutLayer(idbTechCutLayer);
    return kDbSuccess;
  }

  int IdbCheck::initTechCutLayer(IdbLayerCut *layerCut, IdbTechCutLayer *techCutLayer) {
    techCutLayer->set_name(layerCut->get_name());
    techCutLayer->set_type(LayerTypeEnum::kCUT);
    techCutLayer->set_width(layerCut->get_width());
    // TODO(fix laycut spacing, i.e. spacing is a list)
    std::unique_ptr<IdbCutSpacingCheck> idbCutSpacingCheck = std::make_unique<IdbCutSpacingCheck>((*layerCut->get_spacings()[0])
);
    techCutLayer->set_cut_spacing_check(idbCutSpacingCheck);
    initEnclosureCheck(layerCut, techCutLayer);
    initArraySpacingCheck(layerCut, techCutLayer);
    return kDbSuccess;
  }
  int IdbCheck::initArraySpacingCheck(IdbLayerCut *layerCut, IdbTechCutLayer *techCutLayer) {
    IdbLayerCutArraySpacing *arraySpacing                      = layerCut->get_array_spacing();
    std::unique_ptr<IdbArraySpacingCheck> idbArraySpacingCheck = std::make_unique<IdbArraySpacingCheck>();
    idbArraySpacingCheck->set_long_array(arraySpacing->is_long_array());
    idbArraySpacingCheck->set_cut_spacing(arraySpacing->get_cut_spacing());
    vector<IdbArrayCut> arrayCutList = arraySpacing->get_array_cut_list();
    for (auto arrayCut : arrayCutList) {
      idbArraySpacingCheck->add_array_spacing(arrayCut._array_cut, arrayCut._array_spacing);
    }
    techCutLayer->set_array_spacing_check(idbArraySpacingCheck);
    return kDbSuccess;
  }

  int IdbCheck::initEnclosureCheck(IdbLayerCut *layerCut, IdbTechCutLayer *techCutLayer) {
    IdbLayerCutEnclosure *cutEnclosureBelow              = layerCut->get_enclosure_below();
    int belowOverHang1                                   = cutEnclosureBelow->get_overhang_1();
    int belowOverHang2                                   = cutEnclosureBelow->get_overhang_2();
    IdbLayerCutEnclosure *cutEnclosureAbove              = layerCut->get_enclosure_above();
    int aboveOverHang1                                   = cutEnclosureAbove->get_overhang_1();
    int aboveOverHang2                                   = cutEnclosureAbove->get_overhang_2();
    std::unique_ptr<IdbEnclosureCheck> idbEnclosureCheck = std::make_unique<IdbEnclosureCheck>();
    idbEnclosureCheck->setEnclosureBelow(belowOverHang1, belowOverHang2);
    idbEnclosureCheck->setEnclosureAbove(aboveOverHang1, aboveOverHang2);
    techCutLayer->set_enclosure_check(idbEnclosureCheck);
    return kDbSuccess;
  }

  // init viaList
  int IdbCheck::initTechViaList(IdbLayout *layout) {
    IdbVias *idbVias            = layout->get_via_list();
    vector<IdbVia *> idbViaList = idbVias->get_via_list();
    if (idbViaList.size() == 0) {
      return kDbFail;
    }
    for (auto via : idbViaList) {
      IdbViaMaster *viaMaster                        = via->get_instance();
      vector<IdbViaMasterFixed *> viaMasterFixedList = viaMaster->get_master_fixed_list();
      int viaFixedCount                              = 0;
      // need via fixed name!!!!!!
      std::unique_ptr<IdbTechVia> idbTechVia = std::make_unique<IdbTechVia>(via->get_name());
      IdbTechVia *techVia                    = idbTechVia.get();
      for (auto viaMasterFixed : viaMasterFixedList) {
        if (viaFixedCount == 0) {
          IdbRect *rect         = viaMasterFixed->get_rect(0);
          IdbTechRect *techRect = techVia->get_bottom_layer_shape();
          initViaMetalRect(rect, techRect);
        } else if (viaFixedCount == 2) {
          IdbRect *rect         = viaMasterFixed->get_rect(0);
          IdbTechRect *techRect = techVia->get_top_layer_shape();
          initViaMetalRect(rect, techRect);
        } else {
          IdbTechCutLayer *cutLayer = _tech->getCutLayer(viaMasterFixed->get_layer()->get_name());
          int cutLayerId            = cutLayer->get_layer_id();
          techVia->set_cut_layer_id(cutLayerId);
          techVia->set_bottom_layer_default_direction(_tech->getRoutingLayer(cutLayerId)->get_direction());
          techVia->set_top_layer_default_direction(_tech->getRoutingLayer(cutLayerId + 1)->get_direction());
          initViaCutRectList(viaMasterFixed, techVia);
        }
        ++viaFixedCount;
      }
      _tech->addVia(idbTechVia);
    }
    return kDbSuccess;
  }

  int IdbCheck::initViaMetalRect(IdbRect *rect, IdbTechRect *techRect) {
    int llx = rect->get_low_x();
    int lly = rect->get_low_y();
    int urx = rect->get_high_x();
    int ury = rect->get_high_y();
    techRect->setRectPoint(llx, lly, urx, ury);
    return kDbSuccess;
  }
  int IdbCheck::initViaCutRectList(IdbViaMasterFixed *viaMasterFixed, IdbTechVia *techVia) {
    vector<IdbRect *> idbRectList = viaMasterFixed->get_rect_list();
    int rectListSize              = idbRectList.size();
    int rectCount                 = 0;
    if (rectListSize == 1) {
      techVia->set_cut_num(ViaCutNumEnum::k1CUTVIA);
    }
    if (rectListSize == 2) {
      techVia->set_cut_num(ViaCutNumEnum::k2CUTVIA);
    }
    for (auto &idbRect : idbRectList) {
      int llx = idbRect->get_low_x();
      int lly = idbRect->get_low_y();
      int urx = idbRect->get_high_x();
      int ury = idbRect->get_high_y();
      if (rectCount == rectListSize - 1) {
        if (urx < 0 && llx < 0) {
          techVia->set_cut_array_type(CutArrayTypeEnum::kWEST);
        } else if (urx > 0 && llx > 0) {
          techVia->set_cut_array_type(CutArrayTypeEnum::kEAST);
        } else if (ury < 0 && lly < 0) {
          techVia->set_cut_array_type(CutArrayTypeEnum::kSOUTH);
        } else if (ury > 0 && lly > 0) {
          techVia->set_cut_array_type(CutArrayTypeEnum::kNORTH);
        }
      }
      techVia->addCutRectList(llx, lly, urx, ury);
    }
    return kDbSuccess;
  }
  // init viaRuleList
  int IdbCheck::initTechViaRuleList(IdbLayout *layout) {
    IdbViaRuleList *idbViaRuleList                   = layout->get_via_rule_list();
    vector<IdbViaRuleGenerate *> viaRuleGenerateList = idbViaRuleList->get_rule_list();
    for (auto viaRuleGenerate : viaRuleGenerateList) {
      std::unique_ptr<IdbTechViaRule> idbTechViaRule = std::make_unique<IdbTechViaRule>(viaRuleGenerate->get_name());
      IdbTechViaRule *techViaRule                    = idbTechViaRule.get();
      int cutSpacingX                                = viaRuleGenerate->get_spacing_x();
      int cutSapcingY                                = viaRuleGenerate->get_spacing_y();
      techViaRule->setCutSpacing(cutSpacingX, cutSapcingY);
      initViaRuleLayer(viaRuleGenerate, techViaRule);
      initViaRuleEnclosure(viaRuleGenerate, techViaRule);
      initViaRuleCutRect(viaRuleGenerate, techViaRule);

      _tech->addViaRule(idbTechViaRule);
    }
    return kDbSuccess;
  }

  int IdbCheck::initViaRuleLayer(IdbViaRuleGenerate *viaRuleGenerate, IdbTechViaRule *techViaRule) {
    std::string bottomLayerName = viaRuleGenerate->get_layer_bottom()->get_name();
    std::string topLayerName    = viaRuleGenerate->get_layer_top()->get_name();
    std::string cutLayerName    = viaRuleGenerate->get_layer_cut()->get_name();
    techViaRule->set_bottom_layer(_tech->getRoutingLayer(bottomLayerName));
    techViaRule->set_top_layer(_tech->getRoutingLayer(topLayerName));
    techViaRule->set_cut_layer(_tech->getCutLayer(cutLayerName));

    return kDbSuccess;
  }

  int IdbCheck::initViaRuleEnclosure(IdbViaRuleGenerate *viaRuleGenerate, IdbTechViaRule *techViaRule) {
    IdbLayerCutEnclosure *enclosureBottom = viaRuleGenerate->get_enclosure_bottom();
    int bottomOverHang1                   = enclosureBottom->get_overhang_1();
    int bottomOverHang2                   = enclosureBottom->get_overhang_2();
    techViaRule->set_bottom_enclosure(bottomOverHang1, bottomOverHang2);

    IdbLayerCutEnclosure *enclosureTop = viaRuleGenerate->get_enclosure_top();
    int topOverHang1                   = enclosureTop->get_overhang_1();
    int topOverHang2                   = enclosureTop->get_overhang_2();
    techViaRule->set_top_enclosure(topOverHang1, topOverHang2);

    return kDbSuccess;
  }
  int IdbCheck::initViaRuleCutRect(IdbViaRuleGenerate *viaRuleGenerate, IdbTechViaRule *techViaRule) {
    IdbRect *rect = viaRuleGenerate->get_cut_rect();
    int llx       = rect->get_low_x();
    int lly       = rect->get_low_y();
    int urx       = rect->get_high_x();
    int ury       = rect->get_high_y();
    techViaRule->setCutRect(llx, lly, urx, ury);
    return kDbSuccess;
  }

}  // namespace idb
