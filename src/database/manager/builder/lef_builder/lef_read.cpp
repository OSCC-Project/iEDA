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
 * @project		iDB
 * @file		lef_read.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        There is a lef builder to build data structure from lef.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lef_read.h"

#include <algorithm>
#include <cmath>

#include "IdbLayer.h"
#include "IdbLayerShape.h"
#include "IdbLayout.h"
#include "IdbObs.h"
#include "Str.hh"
#include "property_parser/cutlayer_parser.h"
#include "property_parser/masterslicelayer_parser.h"
#include "property_parser/routinglayer_parser.h"

using std::string;
using std::vector;

namespace idb {

LefRead::LefRead(IdbLefService* lef_service)
{
  _lef_service = lef_service;
  _this_cell_master = nullptr;
}

LefRead::~LefRead()
{
}

bool LefRead::check_type(lefrCallbackType_e type)
{
  if (type >= 0 && type <= lefrLibraryEndCbkType) {
    return true;
  } else {
    std::cout << "Error lefrCallbackType_e = " << type << std::endl;
    return false;
  }
}

bool LefRead::createDb(const char* file_name)
{
  // lefrSetAntennaInputCbk(antennaCB);
  // lefrSetAntennaInoutCbk(antennaCB);
  // lefrSetAntennaOutputCbk(antennaCB);
  // lefrSetArrayBeginCbk(arrayBeginCB);
  // lefrSetArrayCbk(arrayCB);
  // lefrSetArrayEndCbk(arrayEndCB);
  // lefrSetBusBitCharsCbk(busBitCharsCB);
  // lefrSetCaseSensitiveCbk(caseSensCB);
  // lefrSetClearanceMeasureCbk(clearanceCB);
  // lefrSetDividerCharCbk(dividerCB);
  // lefrSetNoWireExtensionCbk(noWireExtCB);
  // lefrSetNoiseMarginCbk(noiseMarCB);
  // lefrSetEdgeRateThreshold1Cbk(edge1CB);
  // lefrSetEdgeRateThreshold2Cbk(edge2CB);
  // lefrSetEdgeRateScaleFactorCbk(edgeScaleCB);
  // lefrSetNoiseTableCbk(noiseTableCB);
  // lefrSetCorrectionTableCbk(correctionCB);
  // lefrSetDielectricCbk(dielectricCB);
  // lefrSetIRDropBeginCbk(irdropBeginCB);
  // lefrSetIRDropCbk(irdropCB);
  // lefrSetIRDropEndCbk(irdropEndCB);
  lefrSetLayerCbk(layerCB);
  // lefrSetLibraryEndCbk(doneCB);
  lefrSetMacroBeginCbk(macroBeginCB);
  lefrSetMacroCbk(macroCB);
  lefrSetMacroEndCbk(macroEndCB);
  lefrSetManufacturingCbk(manufacturingCB);
  lefrSetMaxStackViaCbk(maxStackViaCB);
  // lefrSetMinFeatureCbk(minFeatureCB);
  lefrSetNonDefaultCbk(nonDefaultCB);
  lefrSetObstructionCbk(obstructionCB);
  lefrSetPinCbk(pinCB);
  // lefrSetPropBeginCbk(propDefBeginCB);
  lefrSetPropCbk(propDefCB);
  // lefrSetPropEndCbk(propDefEndCB);
  lefrSetSiteCbk(siteCB);
  // lefrSetSpacingBeginCbk(spacingBeginCB);
  // lefrSetSpacingCbk(spacingCB);
  // lefrSetSpacingEndCbk(spacingEndCB);
  // lefrSetTimingCbk(timingCB);
  lefrSetUnitsCbk(unitsCB);
  // lefrSetUseMinSpacingCbk(useMinSpacingCB);
  // lefrSetVersionCbk(versionCB);
  lefrSetViaCbk(viaCB);
  lefrSetViaRuleCbk(viaRuleCB);
  // lefrSetInputAntennaCbk(antennaCB);
  // lefrSetOutputAntennaCbk(antennaCB);
  // lefrSetInoutAntennaCbk(antennaCB);
  // lefrSetLogFunction(errorCB);
  // lefrSetWarningLogFunction(warningCB);
  // lefrSetLineNumberFunction(lineNumberCB);

  // Available callbacks not registered - FIXME??
  // lefrSetDensityCbk
  // lefrSetExtensionCbk
  // lefrSetFixedMaskCbk
  // lefrSetMacroClassTypeCbk
  // lefrSetMacroFixedMaskCbk
  // lefrSetMacroForeignCbk
  // lefrSetMacroOriginCbk
  // lefrSetMacroSiteCbk
  // lefrSetMacroSizeCbk

  lefrSetDeltaNumberLines(1000000);
  lefrInit();

  FILE* file = fopen(file_name, "r");

  if (file == NULL) {
    std::cout << "Open lef file failed..." << std::endl;
    return false;
  }

  _file_name = file_name;

  int res = lefrRead(file, file_name, (void*) this);

  fclose(file);

  lefrClear();

  if (res)
    return false;

  return true;
}

int LefRead::manufacturingCB(lefrCallbackType_e c, double lef_num, lefiUserData data)
{
  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Manufacturing Grid] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_manufacture_grid(lef_num);
  return kDbSuccess;
}

int LefRead::parse_manufacture_grid(double value)
{
  IdbLayout* layout = _lef_service->get_layout();
  layout->set_manufacture_grid(transUnitDB(value));

  return kDbSuccess;
}

int LefRead::propDefCB(lefrCallbackType_e c, lefiProp* prop, lefiUserData data)
{
  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : property definition] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_property_definition(prop);
  return kDbSuccess;
}

int LefRead::parse_property_definition(lefiProp* prop)
{
  IdbLayout* layout = _lef_service->get_layout();

  auto property_type = prop->lefiProp::propType();
  std::string name = prop->lefiProp::propName();
  auto data_type = prop->lefiProp::dataType();

  // set max via stack
  if (name == "LEF58_MAXVIASTACK" && prop->lefiProp::hasString()) {
    std::string data = prop->lefiProp::string();
    parse_max_stack_via_lef58(data);
  }

  return kDbSuccess;
}

int LefRead::parse_max_stack_via_lef58(std::string data)
{
  IdbLayout* layout = _lef_service->get_layout();

  auto max_via_stack = new IdbMaxViaStack();

  const char* sep = " ";
  vector<std::string> strs = ieda::Str::split(data.c_str(), sep);
  for (size_t i = 0; i < strs.size(); ++i) {
    if (strs[i] == "MAXVIASTACK") {
      auto number = atoi(strs[i + 1].c_str());
      max_via_stack->set_stacked_via_num(number);
    }

    if (strs[i] == "NOSINGLE") {
      max_via_stack->set_no_single(true);
    }

    if (strs[i] == "RANGE") {
      std::string bottom_layer = strs[i + 1];
      std::string top_layer = strs[i + 2];
      max_via_stack->set_layer_bottom(bottom_layer);
      max_via_stack->set_layer_top(top_layer);
    }
  }

  layout->set_max_via_stack(max_via_stack);

  return kDbSuccess;
}

int LefRead::maxStackViaCB(lefrCallbackType_e c, lefiMaxStackVia* maxStack, lefiUserData data)
{
  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : max via stack] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_max_stack_via(maxStack);
  return kDbSuccess;
}

int LefRead::parse_max_stack_via(lefiMaxStackVia* maxStack)
{
  IdbLayout* layout = _lef_service->get_layout();

  auto max_via_stack = new IdbMaxViaStack();
  max_via_stack->set_stacked_via_num(maxStack->lefiMaxStackVia::maxStackVia());
  std::string top_layer = maxStack->lefiMaxStackVia::maxStackViaTopLayer();
  max_via_stack->set_layer_top(top_layer);
  std::string bottom_layer = maxStack->lefiMaxStackVia::maxStackViaBottomLayer();
  max_via_stack->set_layer_bottom(bottom_layer);

  layout->set_max_via_stack(max_via_stack);

  return kDbSuccess;
}

int LefRead::siteCB(lefrCallbackType_e c, lefiSite* lef_site, lefiUserData data)
{
  if (lef_site == nullptr) {
    std::cout << "Sites is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Site] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_sites(lef_site);
  return kDbSuccess;
}

int LefRead::parse_sites(lefiSite* lef_site)
{
  if (lef_site == nullptr) {
    std::cout << "Site is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();
  IdbSites* sites = layout->get_sites();

  string site_name = lef_site->name();
  IdbSite* site = sites->add_site_list(site_name);

  // site->set_name(const_cast<char *>(lef_site->lefiSite::name()));

  if (lef_site->hasXSymmetry()) {
    site->set_symmetry(IdbSymmetry::kX);
  } else if (lef_site->hasYSymmetry()) {
    site->set_symmetry(IdbSymmetry::kY);
  } else if (lef_site->has90Symmetry()) {
    site->set_symmetry(IdbSymmetry::kR90);
  } else {
    site->set_symmetry(IdbSymmetry::kNone);
  }

  if (lef_site->hasSize()) {
    site->set_width(transUnitDB(lef_site->sizeX()));
    site->set_height(transUnitDB(lef_site->sizeY()));
  }

  site->set_class(lef_site->siteClass());

  return kDbSuccess;
}

int LefRead::unitsCB(lefrCallbackType_e c, lefiUnits* lef_unit, lefiUserData data)
{
  if (lef_unit == nullptr) {
    std::cout << "Units is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Units] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_units(lef_unit);
  return kDbSuccess;
}

int LefRead::parse_units(lefiUnits* lef_units)
{
  if (lef_units == nullptr) {
    std::cout << "Units is nullPtr..." << std::endl;

    return kDbFail;
  }
  IdbLayout* layout = _lef_service->get_layout();
  if (layout->get_units() != nullptr) {
    std::cout << "Tech Units has been init, ignore this lef units..." << std::endl;
    return kDbSuccess;
  }

  IdbUnits* this_units = new IdbUnits();
  layout->set_units(this_units);

  if (lef_units->hasTime()) {
    double time = lef_units->time();
    this_units->set_nanoseconds(static_cast<int32_t>(time));
  }

  if (lef_units->hasCapacitance()) {
    double capacitance = lef_units->capacitance();
    this_units->set_picofarads(static_cast<int32_t>(capacitance));
  }

  if (lef_units->hasResistance()) {
    double resistance = lef_units->resistance();
    this_units->set_ohms(static_cast<int32_t>(resistance));
  }

  if (lef_units->hasPower()) {
    double power = lef_units->power();
    this_units->set_milliwatts(static_cast<int32_t>(power));
  }

  if (lef_units->hasCurrent()) {
    double current = lef_units->current();
    this_units->set_milliamps(static_cast<int32_t>(current));
  }

  if (lef_units->hasVoltage()) {
    double voltage = lef_units->voltage();
    this_units->set_volts(static_cast<int32_t>(voltage));
  }

  if (lef_units->hasDatabase()) {
    double databaseNumber = lef_units->databaseNumber();
    this_units->set_microns_dbu(static_cast<int32_t>(databaseNumber));
  }

  if (lef_units->hasFrequency()) {
    double frequency = lef_units->frequency();
    this_units->set_megahertz(static_cast<int32_t>(frequency));
  }

  //   units->print();
  return kDbSuccess;
}

int LefRead::layerCB(lefrCallbackType_e c, lefiLayer* lef_layer, lefiUserData data)
{
  if (lef_layer == nullptr) {
    std::cout << "Layer is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Layer] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_layer(lef_layer);

  return kDbSuccess;
}

int LefRead::parse_layer(lefiLayer* lef_layer)
{
  if (lef_layer == nullptr) {
    std::cout << "Layer is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();
  IdbLayers* layers = layout->get_layers();
  if (layers->find_layer(lef_layer->name()) != nullptr) {
    std::cout << "Warning, layer is exist, name = " << lef_layer->name() << std::endl;
    return kDbFail;
  }

  IdbLayer* layer = nullptr;

  if (lef_layer->hasType()) {
    layer = layers->set_layer(lef_layer->name(), lef_layer->type());
    // layer->set_type(lef_layer->type());
  }

  if (layer == nullptr) {
    std::cout << "Layer is nullPtr..." << std::endl;
    return kDbFail;
  }

  if (IdbLayerType::kLayerCut == layer->get_type()) {
    layers->add_cut_layer(layer);
    IdbLayerCut* layer_cut = dynamic_cast<IdbLayerCut*>(layer);
    parse_layer_cut(lef_layer, layer_cut);
  } else if (IdbLayerType::kLayerRouting == layer->get_type()) {
    layers->add_routing_layer(layer);
    IdbLayerRouting* layer_routing = dynamic_cast<IdbLayerRouting*>(layer);
    parse_layer_routing(lef_layer, layer_routing);
  } else if (IdbLayerType::kLayerMasterslice == layer->get_type()) {
    IdbLayerMasterslice* layer_masterslice = dynamic_cast<IdbLayerMasterslice*>(layer);
    parse_layer_masterslice(lef_layer, layer_masterslice);
  } else if (IdbLayerType::kLayerOverlap == layer->get_type()) {
    IdbLayerOverlap* layer_overlap = dynamic_cast<IdbLayerOverlap*>(layer);
    parse_layer_overlap(lef_layer, layer_overlap);
  } else if (IdbLayerType::kLayerImplant == layer->get_type()) {
    IdbLayerImplant* layer_implant = dynamic_cast<IdbLayerImplant*>(layer);
    parse_layer_implant(lef_layer, layer_implant);
  } else {
    //!---tbd------------
  }

  // if(lef_layer->hasWidth())
  //     layer->

  //!---tbd------------

  //   layer->print();

  // std::cout << "Parse Layer success..." << std::endl;
  return kDbSuccess;
}

int LefRead::parse_layer_cut(lefiLayer* lef_layer, IdbLayerCut* layer_cut)
{
  if (lef_layer == nullptr || layer_cut == nullptr) {
    return kDbFail;
  }

  // width
  if (lef_layer->hasWidth()) {
    layer_cut->set_width(transUnitDB(lef_layer->width()));
  }

  // spacing
  int num_spacing = lef_layer->numSpacing();
  for (int i = 0; i < num_spacing; ++i) {
    int32_t cut_spacing = transUnitDB(lef_layer->spacing(i));
    IdbLayerCutSpacing* spacing = new IdbLayerCutSpacing(cut_spacing);
    if (lef_layer->hasSpacingAdjacent(i)) {
      int adj_cuts = lef_layer->spacingAdjacentCuts(i);
      double cut_within = lef_layer->spacingAdjacentWithin(i);
      spacing->set_adjacent_cuts(IdbLayerCutSpacing::AdjacentCuts(adj_cuts, transUnitDB(cut_within)));
    }
    layer_cut->add_spacing(spacing);
  }

  for (int i = 0; i < lef_layer->numEnclosure(); i++) {
    //!<------------tbd-------------------
    if (lef_layer->hasEnclosureRule(i)) {
      string enclosure_name = lef_layer->enclosureRule(i);
      if (enclosure_name.compare("ABOVE") == 0) {
        IdbLayerCutEnclosure* enclosure = layer_cut->get_enclosure_above();
        enclosure->set_overhang_1(transUnitDB(lef_layer->enclosureOverhang1(i)));
        enclosure->set_overhang_2(transUnitDB(lef_layer->enclosureOverhang2(i)));
      } else if (enclosure_name.compare("BELOW") == 0) {
        IdbLayerCutEnclosure* enclosure = layer_cut->get_enclosure_below();
        enclosure->set_overhang_1(transUnitDB(lef_layer->enclosureOverhang1(i)));
        enclosure->set_overhang_1(transUnitDB(lef_layer->enclosureOverhang2(i)));
      }
    }
  }

  // array spacing
  if (lef_layer->hasArraySpacing()) {
    IdbLayerCutArraySpacing* array_spacing = layer_cut->get_array_spacing();
    if (lef_layer->hasLongArray()) {
      array_spacing->set_long_array(true);
    }

    array_spacing->set_cut_spacing(transUnitDB(lef_layer->cutSpacing()));
    array_spacing->set_array_cut_num(lef_layer->numArrayCuts());

    for (int i = 0; i < lef_layer->numArrayCuts(); ++i) {
      array_spacing->set_array_value(i, lef_layer->arrayCuts(i), transUnitDB(lef_layer->arraySpacing(i)));
    }
  }

  for (int i = 0; i < lef_layer->numProps(); i++) {
    CutLayerParser cutlayer_parser(_lef_service);
    cutlayer_parser.parse(lef_layer->propName(i), lef_layer->propValue(i), layer_cut);
  }

  return kDbSuccess;
}

int LefRead::parse_layer_routing(lefiLayer* lef_layer, IdbLayerRouting* layer_routing)
{
  if (lef_layer == nullptr || layer_routing == nullptr) {
    return kDbFail;
  }

  // width
  if (lef_layer->hasWidth()) {
    layer_routing->set_width(transUnitDB(lef_layer->width()));
  }

  // min width
  // MINWIDTH width
  // Specifies the minimum legal object width on the routing layer. For example, MINWIDTH
  // 0.15 specifies that the width of every object must be greater than or equal to 0.15 μm.
  // This value is used for verification purposes, and does not affect the routing width. The
  // WIDTH statement defines the default routing width on the layer.
  // Default: The value of the WIDTH statement
  // Type: Float, specified in microns
  if (lef_layer->hasMinwidth()) {
    layer_routing->set_min_width(transUnitDB(lef_layer->minwidth()));
  } else {
    // Default value
    layer_routing->set_min_width(layer_routing->get_width());
  }

  // max width
  if (lef_layer->hasMaxwidth()) {
    layer_routing->set_max_width(transUnitDB(lef_layer->maxwidth()));
  }

  // direction
  if (lef_layer->hasDirection()) {
    layer_routing->set_direction(lef_layer->direction());
  }

  // definition in LEF file: OFFSET {distance | xDistance yDistance}
  // definition in LEF file: PITCH {distance | xDistance yDistance}
  // If Specifies the value that is used for the preferred directionrouting tracks,
  //   set orient_x = orient_y = offset
  // else if Specifies the x for vertical routing tracks, and the y for horizontal routing tracks.
  //   set orient_x & orient_y seperate

  // pitch
  IdbLayerOrientValue pitch;
  if (lef_layer->hasPitch()) {
    pitch.type = IdbLayerOrientType::kBothXY;
    pitch.orient_x = transUnitDB(lef_layer->pitch());
    pitch.orient_y = transUnitDB(lef_layer->pitch());
    layer_routing->set_pitch(pitch);
  } else if (lef_layer->hasXYPitch()) {
    pitch.type = IdbLayerOrientType::kSeperateXY;
    pitch.orient_x = transUnitDB(lef_layer->pitchX());
    pitch.orient_y = transUnitDB(lef_layer->pitchY());
    layer_routing->set_pitch(pitch);
  } else {
    // nothing to do
  }

  // offset
  IdbLayerOrientValue offset;
  if (lef_layer->hasOffset()) {
    offset.type = IdbLayerOrientType::kBothXY;
    offset.orient_x = transUnitDB(lef_layer->offset());
    offset.orient_y = transUnitDB(lef_layer->offset());
    layer_routing->set_offset(offset);
  } else if (lef_layer->hasXYOffset()) {
    offset.type = IdbLayerOrientType::kSeperateXY;
    offset.orient_x = transUnitDB(lef_layer->offsetX());
    offset.orient_y = transUnitDB(lef_layer->offsetY());
    layer_routing->set_offset(offset);
  } else {
    // For best routing results, most standard cells have a 1/2 pitch offset between the MACRO SIZE boundary and the
    // center of cell pins that should be aligned with the routing grid.
    offset.type = IdbLayerOrientType::kBothXY;
    offset.orient_x = layer_routing->get_pitch_x() / 2;
    offset.orient_y = layer_routing->get_pitch_y() / 2;
    layer_routing->set_offset(offset);
  }

  // wire extension
  // Specifies the distance by which wires are extended at vias. You must specify a value that is more than half of the
  // routing width. Default: Wires are extended half of the routing width
  if (lef_layer->hasWireExtension()) {
    int32_t wire_extension = transUnitDB(lef_layer->wireExtension());
    wire_extension = wire_extension < (layer_routing->get_width() / 2) ? (layer_routing->get_width() / 2) : wire_extension;
    layer_routing->set_wire_extension(wire_extension);
  }

  // thickness
  if (lef_layer->hasThickness()) {
    layer_routing->set_thickness(transUnitDB(lef_layer->thickness()));
  }

  // height
  if (lef_layer->hasHeight()) {
    layer_routing->set_height(transUnitDB(lef_layer->height()));
  }

  // resistance
  if (lef_layer->hasResistance()) {
    layer_routing->set_resistance(lef_layer->resistance());
  }

  // capacitance
  if (lef_layer->hasCapacitance()) {
    layer_routing->set_capacitance(lef_layer->capacitance());
  }

  // edge capacitance
  if (lef_layer->hasEdgeCap()) {
    layer_routing->set_edge_capacitance(lef_layer->edgeCap());
  }

  // spacing
  if (lef_layer->hasSpacingNumber()) {
    int spacing_num = lef_layer->numSpacing();
    for (int i = 0; i < spacing_num; i++) {
      IdbLayerSpacingList* spacing_list = layer_routing->get_spacing_list();
      IdbLayerSpacing* layer_spacing = new IdbLayerSpacing();
      layer_spacing->set_spacing_type(IdbLayerSpacingType::kSpacingDefault);
      layer_spacing->set_min_spacing(transUnitDB(lef_layer->spacing(i)));

      // RANGE minWidth maxWidth
      if (lef_layer->hasSpacingRange(i)) {
        layer_spacing->set_spacing_type(IdbLayerSpacingType::kSpacingRange);
        layer_spacing->set_min_width(transUnitDB(lef_layer->spacingRangeMin(i)));
        layer_spacing->set_max_width(transUnitDB(lef_layer->spacingRangeMax(i)));
      }
      if (lef_layer->hasSpacingNotchLength(i)) {
        auto& spacing_notch = layer_routing->get_spacing_notchlength();
        spacing_notch.set_notch_length(transUnitDB(lef_layer->spacingNotchLength(i)));
        spacing_notch.set_min_spacing(transUnitDB(lef_layer->spacing(i)));
      }
      spacing_list->add_spacing(layer_spacing);
    }
  }

  // spacingtable
  if (lef_layer->numSpacingTable() > 0) {
    int num_spacingtable = lef_layer->numSpacingTable();
    for (int i = 0; i < num_spacingtable; ++i) {
      auto* i_spacingtbl = lef_layer->spacingTable(i);
      if (i_spacingtbl->isParallel()) {
        auto* i_parallel = i_spacingtbl->parallel();
        int num_width = i_parallel->numWidth();
        int num_length = i_parallel->numLength();
        auto parallel = std::make_shared<IdbParallelSpacingTable>(num_width, num_length);
        for (int j = 0; j < num_length; ++j) {
          parallel->set_parallel_length(j, transUnitDB(i_parallel->length(j)));
        }
        for (int j = 0; j < num_width; ++j) {
          parallel->set_width(j, transUnitDB(i_parallel->width(j)));
          for (int k = 0; k < num_length; ++k) {
            parallel->set_spacing(j, k, transUnitDB(i_parallel->widthSpacing(j, k)));
          }
        }
        layer_routing->set_parallel_spacing_table(parallel);
      }
    }
  }

  if (lef_layer->hasArea()) {
    layer_routing->set_area(transAreaDB(lef_layer->area()));
  }

  // MINSTEP
  if (lef_layer->hasMinstep()) {
    lef_layer->minstep(0);
    auto minstep = std::make_shared<IdbMinStep>(transUnitDB(lef_layer->minstep(0)));
    if (lef_layer->hasMinstepType(0)) {
      minstep->set_type(lef_layer->minstepType(0));
    }
    if (lef_layer->hasMinstepLengthsum(0)) {
      minstep->set_max_length(transUnitDB(lef_layer->minstepLengthsum(0)));
    }
    if (lef_layer->hasMinstepMaxedges(0)) {
      minstep->set_max_edges(lef_layer->minstepMaxedges(0));
    }
    layer_routing->set_min_step(minstep);
  }

  // MINENCLOSEDAREA
  int32_t min_area_num = lef_layer->numMinenclosedarea();
  IdbMinEncloseAreaList* min_enclose_area = layer_routing->get_min_enclose_area_list();
  for (int i = 0; i < min_area_num; ++i) {
    int32_t area = transAreaDB(lef_layer->minenclosedarea(i));
    int32_t width = -1;
    if (lef_layer->hasMinenclosedareaWidth(i)) {
      width = transUnitDB(lef_layer->minenclosedareaWidth(i));
    }

    min_enclose_area->add_min_area(area, width);
  }

  // MINIMUMDENSITY
  if (lef_layer->hasMinimumDensity()) {
    layer_routing->set_min_density(lef_layer->minimumDensity());
  }

  // MAXIMUMDENSITY
  if (lef_layer->hasMaximumDensity()) {
    layer_routing->set_max_density(lef_layer->maximumDensity());
  }

  // DENSITYCHECKWINDOW
  if (lef_layer->hasDensityCheckWindow()) {
    layer_routing->set_density_check_length(lef_layer->densityCheckWindowLength());
    layer_routing->set_density_check_width(lef_layer->densityCheckWindowWidth());
  }
  // DENSITYCHECKSTEP
  if (lef_layer->hasDensityCheckStep()) {
    layer_routing->set_density_check_step(lef_layer->densityCheckStep());
  }

  // MINIMUMCUT
  // layer_routing->set_min_cut_num(lef_layer->numMinimumcut() ));
  // layer_routing->set_min_cut_width(lef_layer->numMinimu)

  for (int i = 0; i < lef_layer->numProps(); i++) {
    RoutingLayerParser routing_layer_parser(_lef_service);
    routing_layer_parser.parse(lef_layer->propName(i), lef_layer->propValue(i), layer_routing);
  }

  return kDbSuccess;
}

int LefRead::parse_layer_masterslice(lefiLayer* lef_layer, IdbLayerMasterslice* layer_master)
{
  if (lef_layer == nullptr || layer_master == nullptr) {
    return kDbFail;
  }
  for (int i = 0; i < lef_layer->numProps(); i++) {
    MastersliceLayerParser masterslice_parser(_lef_service);
    masterslice_parser.parse(lef_layer->propName(i), lef_layer->propValue(i), layer_master);
  }
  return kDbSuccess;
}

int LefRead::parse_layer_overlap(lefiLayer* lef_layer, IdbLayerOverlap* layer_overlap)
{
  if (lef_layer == nullptr || layer_overlap == nullptr) {
    return kDbFail;
  }

  return kDbSuccess;
}

int LefRead::parse_layer_implant(lefiLayer* lef_layer, IdbLayerImplant* layer_implant)
{
  if (lef_layer == nullptr || layer_implant == nullptr) {
    return kDbFail;
  }

  if (lef_layer->hasSpacingNumber()) {
    auto min_spacing_list = layer_implant->get_min_spacing_list();
    for (int i = 0; i < lef_layer->numSpacing(); i++) {
      auto min_spacing = min_spacing_list->add_min_spacing();
      min_spacing->set_min_spacing(transUnitDB(lef_layer->spacing(i)));

      if (lef_layer->hasSpacingName(i)) {
        IdbLayers* layers = _lef_service->get_layout()->get_layers();
        auto layer_2nd = layers->find_layer(lef_layer->spacingName(i));
        min_spacing->set_layer_2nd(layer_2nd);
      }
    }
  }

  if (lef_layer->hasWidth()) {
    layer_implant->set_min_width(transUnitDB(lef_layer->width()));
  }

  return kDbSuccess;
}

int LefRead::macroBeginCB(lefrCallbackType_e c, const char* lef_name, lefiUserData data)
{
  if (lef_name == nullptr) {
    // std::cout << "Macro is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Macro] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_macro_new(lef_name);

  return kDbSuccess;
}

int LefRead::parse_macro_new(const char* macro_name)
{
  if (macro_name == nullptr) {
    // std::cout << "Macro is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();
  IdbCellMasterList* master_list = layout->get_cell_master_list();

  if (nullptr != master_list->find_cell_master(macro_name)) {
    _this_cell_master = nullptr;
    std::cout << "[idb warning] Macro is exist, name = " << macro_name << std::endl;
    return kDbFail;
  }

  IdbCellMaster* cell_master = master_list->set_cell_master(macro_name);

  _this_cell_master = cell_master;

  return kDbSuccess;
}

int LefRead::macroCB(lefrCallbackType_e c, lefiMacro* lef_macro, lefiUserData data)
{
  if (lef_macro == nullptr) {
    // std::cout << "Macro is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Macro] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_macro(lef_macro);
  return kDbSuccess;
}

int LefRead::parse_macro(lefiMacro* lef_macro)
{
  if (lef_macro == nullptr || _this_cell_master == nullptr) {
    // std::cout << "Macro is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();

  if (lef_macro->hasClass()) {
    _this_cell_master->set_type(lef_macro->macroClass());
  }

  if (lef_macro->hasXSymmetry()) {
    _this_cell_master->set_symmetry_x(true);
  }

  if (lef_macro->hasYSymmetry()) {
    _this_cell_master->set_symmetry_y(true);
  }

  if (lef_macro->has90Symmetry()) {
    _this_cell_master->set_symmetry_R90(true);
  }

  if (lef_macro->hasOrigin()) {
    _this_cell_master->set_origin_x(transUnitDB(lef_macro->originX()));
    _this_cell_master->set_origin_y(transUnitDB(lef_macro->originY()));
  }

  if (lef_macro->hasSize()) {
    _this_cell_master->set_width(transUnitDB(lef_macro->sizeX()));
    _this_cell_master->set_height(transUnitDB(lef_macro->sizeY()));
  }

  if (lef_macro->hasSiteName()) {
    auto site_list = layout->get_sites();
    auto* site = site_list->find_site(lef_macro->siteName());
    if (site != nullptr) {
      _this_cell_master->set_site(site);
    }
  }

  // std::cout << "Parse Macro success... Macro name = " << _this_cell_master->get_name() << std::endl;
  return kDbSuccess;
}

int LefRead::macroEndCB(lefrCallbackType_e c, const char* lef_name, lefiUserData data)
{
  if (lef_name == nullptr) {
    // std::cout << "Macro is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : MacroEnd] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_macro_reset(lef_name);

  return kDbSuccess;
}

int LefRead::parse_macro_reset(const char* name)
{
  if (_this_cell_master != nullptr && _this_cell_master->get_name().compare(name) == 0) {
    _this_cell_master = nullptr;
  }

  return kDbSuccess;
}

int LefRead::pinCB(lefrCallbackType_e c, lefiPin* lef_pin, lefiUserData data)
{
  if (lef_pin == nullptr) {
    std::cout << "Pin is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Pin] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_pin(lef_pin);
  return kDbSuccess;
}

int LefRead::parse_pin(lefiPin* lef_pin)
{
  if (lef_pin == nullptr || _this_cell_master == nullptr) {
    // std::cout << "Pin is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();
  IdbLayers* layer_list = layout->get_layers();

  IdbTerm* term = _this_cell_master->add_term(lef_pin->name());
  term->set_as_instance_pin();

  if (lef_pin->hasDirection()) {
    term->set_direction(lef_pin->direction());
  }

  if (lef_pin->hasUse()) {
    term->set_type(lef_pin->use());
  }

  if (lef_pin->hasShape()) {
    term->set_shape(lef_pin->shape());
  }

  // Calculate average coordinate of all the ports
  int32_t coordinate_x = 0;
  int32_t coordinate_y = 0;
  int32_t ll_x = INT_MAX;
  int32_t ll_y = INT_MAX;
  int32_t ur_x = INT_MIN;
  int32_t ur_y = INT_MIN;

  int point_num = 0;

  int port_num = lef_pin->numPorts();
  for (int i = 0; i < port_num; i++) {
    IdbPort* port = term->add_port();
    IdbLayerShape* shape = nullptr;

    lefiGeometries* lef_geometry = lef_pin->port(i);
    int item_num = lef_geometry->numItems();
    for (int j = 0; j < item_num; j++) {
      // only supper rects in v0.1
      //<----tbd------------
      lefiGeomEnum geom_type = lef_geometry->itemType(j);
      switch (geom_type) {
        case lefiGeomClassE: {
          port->set_port_class(lef_geometry->getClass(j));
          break;
        }
        case lefiGeomLayerE: {
          shape = port->find_layer_shape(lef_geometry->getLayer(j));
          if (shape == nullptr) {
            shape = port->add_layer_shape();
            IdbLayer* layer = layer_list->find_layer(lef_geometry->getLayer(j));
            if (layer != nullptr) {
              shape->set_layer(layer);
            }
          }
          break;
        }
        case lefiGeomRectE: {
          if (shape != nullptr) {
            shape->set_type_rect();
            lefiGeomRect* rect = lef_geometry->getRect(j);
            int32_t rect_ll_x = transUnitDB(rect->xl);
            int32_t rect_ll_y = transUnitDB(rect->yl);
            int32_t rect_ur_x = transUnitDB(rect->xh);
            int32_t rect_ur_y = transUnitDB(rect->yh);
            shape->add_rect(rect_ll_x, rect_ll_y, rect_ur_x, rect_ur_y);

            // calculate bounding box
            ll_x = std::min(ll_x, rect_ll_x);
            ll_y = std::min(ll_y, rect_ll_y);
            ur_x = std::max(ur_x, rect_ur_x);
            ur_y = std::max(ur_y, rect_ur_y);

            // calculate average coodinate of all the ports
            //   int32_t mid_x = (rect_ll_x + rect_ur_x) % 2 == 0 ? (rect_ll_x + rect_ur_x) / 2 : (rect_ll_x + rect_ur_x)
            //   / 2 + 1; int32_t mid_y = (rect_ll_y + rect_ur_y) % 2 == 0 ? (rect_ll_y + rect_ur_y) / 2 : (rect_ll_y +
            //   rect_ur_y) / 2 + 1;
            // int32_t mid_x = (rect_ll_x + rect_ur_x);
            // int32_t mid_y = (rect_ll_y + rect_ur_y);

            coordinate_x = coordinate_x + rect_ll_x + rect_ur_x;
            coordinate_y = coordinate_y + rect_ll_y + rect_ur_y;
            point_num += 2;
          }
          break;
        }
        case lefiGeomWidthE: {
          break;
        }
        case lefiGeomPathE: {
          break;
        }
        case lefiGeomPathIterE: {
          break;
        }
        case lefiGeomRectIterE: {
          break;
        }
        case lefiGeomPolygonE: {
          if (shape != nullptr) {
            shape->set_type_rect();
            lefiGeomPolygon* polygon = lef_geometry->getPolygon(j);
            for (auto rect : polygonToRects(polygon)) {
              shape->add_rect(gtl::xl(rect), gtl::yl(rect), gtl::xh(rect), gtl::yh(rect));
              // calculate bounding box
              ll_x = std::min(ll_x, gtl::xl(rect));
              ll_y = std::min(ll_y, gtl::yl(rect));
              ur_x = std::max(ur_x, gtl::xh(rect));
              ur_y = std::max(ur_y, gtl::yh(rect));

              coordinate_x = coordinate_x + gtl::xl(rect) + gtl::xh(rect);
              coordinate_y = coordinate_y + gtl::yl(rect) + gtl::yh(rect);
              point_num += 2;
            }
          }
          break;
        }
        case lefiGeomViaE: {
          if (!shape) {
            break;
          }
          lefiGeomVia* ivia = lef_geometry->getVia(j);
          auto* vialist = layout->get_via_list();
          auto* via = vialist->find_via(ivia->name);
          if (via == nullptr) {
            std::cerr << "Error, cannot find via " << ivia->name << std::endl;
          } else {
            auto* macro_via = via->clone();
            macro_via->set_coordinate(transUnitDB(ivia->x), transUnitDB(ivia->y));
            port->add_via(macro_via);
          }
          break;
        }
        case lefiGeomViaIterE: {
          break;
        }
        case lefiGeomUnknown:
        case lefiGeomLayerExceptPgNetE:
        case lefiGeomLayerMinSpacingE:
        case lefiGeomLayerRuleWidthE:

        default:
          break;
      }
    }
  }

  if (point_num > 0) {
    term->set_has_port(true);
    term->set_average_position(coordinate_x / point_num, coordinate_y / point_num);
    term->set_bounding_box(ll_x, ll_y, ur_x, ur_y);
  } else {
    term->set_has_port(false);
    return kDbSuccess;
  }

  // std::cout << "Parse lef pin success...Pin name = " << lef_pin->name() << std::endl;

  return kDbSuccess;
}

std::vector<GtlRect> LefRead::polygonToRects(lefiGeomPolygon* polygon)
{
  std::vector<GtlRect> rects;

  std::vector<GtlPoint> points;
  points.reserve(polygon->numPoints);
  for (int j = 0; j < polygon->numPoints; ++j) {
    GtlPoint pt(transUnitDB(polygon->x[j]), transUnitDB(polygon->y[j]));
    points.push_back(pt);
  }

  GtlPolygon90 boost_polygon;

  gtl::set_points(boost_polygon, points.begin(), points.end());
  gtl::get_rectangles(rects, boost_polygon);

  return rects;
}

int LefRead::obstructionCB(lefrCallbackType_e c, lefiObstruction* lef_obs, lefiUserData data)
{
  if (lef_obs == nullptr) {
    std::cout << "Obs is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Obstruction] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_obs(lef_obs);
  return kDbSuccess;
}

int LefRead::parse_obs(lefiObstruction* lef_obs)
{
  if (lef_obs == nullptr || _this_cell_master == nullptr) {
    // std::cout << "Obstruction is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();
  IdbLayers* layer_list = layout->get_layers();

  IdbObs* obs = _this_cell_master->add_obs(nullptr);

  lefiGeometries* lef_geometry = lef_obs->geometries();
  IdbObsLayer* obs_layer = nullptr;

  int item_num = lef_geometry->numItems();
  for (int j = 0; j < item_num; j++) {
    lefiGeomEnum geom_type = lef_geometry->itemType(j);
    switch (geom_type) {
      case lefiGeomLayerE: {
        obs_layer = obs->add_obs_layer(nullptr);
        IdbLayer* layer = layer_list->find_layer(lef_geometry->getLayer(j));
        if (layer == nullptr) {
          break;
        }
        IdbLayerShape* layer_shape = obs_layer->get_shape();
        layer_shape->set_layer(layer);
        break;
      }
      case lefiGeomRectE: {
        IdbLayerShape* layer_shape = obs_layer->get_shape();
        layer_shape->set_type_rect();
        lefiGeomRect* rect = lef_geometry->getRect(j);
        layer_shape->add_rect(transUnitDB(rect->xl), transUnitDB(rect->yl), transUnitDB(rect->xh), transUnitDB(rect->yh));
        break;
      }
      case lefiGeomPolygonE: {
        IdbLayerShape* layer_shape = obs_layer->get_shape();
        if (layer_shape != nullptr) {
          layer_shape->set_type_rect();
          lefiGeomPolygon* polygon = lef_geometry->getPolygon(j);
          for (auto rect : polygonToRects(polygon)) {
            layer_shape->add_rect(gtl::xl(rect), gtl::yl(rect), gtl::xh(rect), gtl::yh(rect));
          }
        }
        break;
      }
      //-----------------tbd---------------------
      default:
        break;
    }
  }

  // std::cout << "Parse lef obs success..." << std::endl;
  return kDbSuccess;
}

int LefRead::viaCB(lefrCallbackType_e c, lefiVia* lef_via, lefiUserData data)
{
  if (lef_via == nullptr) {
    std::cout << "Via is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Via] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_via(lef_via);
  return kDbSuccess;
}

int LefRead::parse_via(lefiVia* lef_via)
{
  if (lef_via == nullptr) {
    std::cout << "Via is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();
  IdbLayers* layer_list = layout->get_layers();
  IdbVias* via_list = layout->get_via_list();
  if (via_list->find_via(lef_via->name()) != nullptr) {
    std::cout << "Warning, Via is exist, name = " << lef_via->name() << std::endl;
    return kDbFail;
  }

  IdbVia* via_instance = via_list->add_via(lef_via->name());
  IdbViaMaster* via_master = via_instance->get_instance();
  via_master->set_name(lef_via->name());

  if (lef_via->hasDefault()) {
    via_master->set_default(true);
  }

  if (lef_via->hasGenerated()) {
    // generate type
    //<--------------------tbd-------------------
    via_master->set_type_generate();
  }

  if (lef_via->hasViaRule()) {
    // IdbViaMasterGenerate* master_gernerate = via_master->get_master_generate();
    //  master_gernerate->set_cut_size();
  } else {
    via_master->set_type_fixed();
    // Fixed via
    int32_t min_x = 0;
    int32_t min_y = 0;
    int32_t max_x = 0;
    int32_t max_y = 0;
    // Fixed type
    for (int i = 0; i < lef_via->numLayers(); i++) {
      IdbViaMasterFixed* master_fixed = via_master->add_fixed(lef_via->layerName(i));
      IdbLayer* layer = layer_list->find_layer(lef_via->layerName(i));
      if (layer != nullptr) {
        master_fixed->set_layer(layer);
        for (int j = 0; j < lef_via->numRects(i); j++) {
          int32_t ll_x = transUnitDB(lef_via->xl(i, j));
          int32_t ll_y = transUnitDB(lef_via->yl(i, j));
          int32_t ur_x = transUnitDB(lef_via->xh(i, j));
          int32_t ur_y = transUnitDB(lef_via->yh(i, j));
          master_fixed->add_rect(ll_x, ll_y, ur_x, ur_y);

          // record the core area of cut
          if (layer->get_type() == IdbLayerType::kLayerCut) {
            min_x = std::min(min_x, ll_x);
            min_y = std::min(min_y, ll_y);
            max_x = std::max(max_x, ur_x);
            max_y = std::max(max_y, ur_y);
          }
        }
      }
    }
    via_master->set_cut_rect(min_x, min_y, max_x, max_y);
    via_master->set_via_shape();

    /// rows and cols
    int32_t num_rows = 1;
    int32_t num_cols = 1;
    if (lef_via->hasRowCol()){
      num_rows = lef_via->numCutRows();
      num_cols = lef_via->numCutCols();
    }
    via_master->set_cut_row_col(num_rows, num_cols);

  }

  return kDbSuccess;
}

int LefRead::viaRuleCB(lefrCallbackType_e c, lefiViaRule* lef_via_rule, lefiUserData data)
{
  if (lef_via_rule == nullptr) {
    std::cout << "Via Rule is nullPtr..." << std::endl;
    return kDbFail;
  }

  LefRead* lef_reader = (LefRead*) data;
  if (!lef_reader->check_type(c)) {
    std::cout << "Check Type Error [Lef : Via Rule] ..." << std::endl;
    return kDbFail;
  }

  lef_reader->parse_via_rule(lef_via_rule);
  return kDbSuccess;
}

int LefRead::parse_via_rule(lefiViaRule* lef_via_rule)
{
  if (lef_via_rule == nullptr) {
    std::cout << "Via Rule is nullPtr..." << std::endl;

    return kDbFail;
  }

  IdbLayout* layout = _lef_service->get_layout();
  IdbViaRuleList* via_rule_list = layout->get_via_rule_list();
  IdbLayers* layer_list = layout->get_layers();
  IdbViaRuleGenerate* via_rule_generate = via_rule_list->add_via_rule_generate(lef_via_rule->name());

  // support generate rule only
  if (lef_via_rule->hasGenerate()) {
    // via_rule_generate->set_generate(true);

    for (int i = 0; i < lef_via_rule->numLayers(); i++) {
      lefiViaRuleLayer* leflay = lef_via_rule->layer(i);
      IdbLayer* layer = layer_list->find_layer(leflay->name());
      if (layer == nullptr) {
        break;
      }

      if (layer->is_routing() && via_rule_generate->get_layer_bottom() == nullptr) {
        via_rule_generate->set_layer_bottom(dynamic_cast<IdbLayerRouting*>(layer));
        IdbLayerCutEnclosure* enclousre = via_rule_generate->get_enclosure_bottom();
        if (leflay->hasEnclosure()) {
          enclousre->set_overhang_1(transUnitDB(leflay->enclosureOverhang1()));
          enclousre->set_overhang_2(transUnitDB(leflay->enclosureOverhang2()));
        }
      } else if (layer->is_routing() && via_rule_generate->get_layer_top() == nullptr) {
        via_rule_generate->set_layer_top(dynamic_cast<IdbLayerRouting*>(layer));
        IdbLayerCutEnclosure* enclousre = via_rule_generate->get_enclosure_top();
        if (leflay->hasEnclosure()) {
          enclousre->set_overhang_1(transUnitDB(leflay->enclosureOverhang1()));
          enclousre->set_overhang_2(transUnitDB(leflay->enclosureOverhang2()));
        }
      } else if (layer->is_cut()) {
        IdbLayerCut* cut_layer = dynamic_cast<IdbLayerCut*>(layer);
        cut_layer->set_via_rule_default(via_rule_generate);
        via_rule_generate->set_layer_cut(dynamic_cast<IdbLayerCut*>(layer));

        if (leflay->hasRect()) {
          int32_t ll_x = transUnitDB(leflay->xl());
          int32_t ll_y = transUnitDB(leflay->yl());
          int32_t ur_x = transUnitDB(leflay->xh());
          int32_t ur_y = transUnitDB(leflay->yh());
          via_rule_generate->set_cut_rect(ll_x, ll_y, ur_x, ur_y);
        }

        if (leflay->hasSpacing()) {
          int32_t spacing_x = transUnitDB(leflay->spacingStepX());
          int32_t spacing_y = transUnitDB(leflay->spacingStepY());
          via_rule_generate->set_spacing(spacing_x, spacing_y);
        }
      } else {
        /// do nothing
      }
    }

    ////swap routing layer of bottom and top if need
    via_rule_generate->swap_routing_layer();
  }

  return kDbSuccess;
}

int LefRead::nonDefaultCB(lefrCallbackType_e c, lefiNonDefault* def_nd, lefiUserData data)
{
  if (def_nd == nullptr) {
    std::cout << "NonDefault Rule is nullPtr..." << std::endl;
    return kDbFail;
  }

  return kDbSuccess;
}
}  // namespace idb
