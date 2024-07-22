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

#include "timing_engine_builder.h"

#include "../data_manager/data_manager.h"
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "idm.h"
#include "timing_engine_util.h"

namespace ito {

TimingEngineBuilder::TimingEngineBuilder()
{
}

TimingEngineBuilder::~TimingEngineBuilder()
{
}

void TimingEngineBuilder::buildEngine()
{
  initISTA();
  findEquivLibCells();
  findDrvrVertices();
  findBufferCells();
  calcCellTargetLoads();
}

void TimingEngineBuilder::findEquivLibCells()
{
  vector<LibLibrary*> equiv_libs;
  auto& all_libs = timingEngine->get_sta_engine()->getAllLib();
  for (auto& lib : all_libs) {
    for (auto& cell : lib->get_cells()) {
      if (timingEngine->canFindLibertyCell(cell.get())) {
        equiv_libs.push_back(lib.get());
        break;
      }
    }
  }

  timingEngine->get_sta_engine()->makeClassifiedCells(equiv_libs);
}

void TimingEngineBuilder::initISTA()
{
  ista::TimingEngine::destroyTimingEngine();

  auto timing_engine = ista::TimingEngine::getOrCreateTimingEngine();

  const char* design_work_space = toConfig->get_design_work_space().c_str();
  vector<const char*> lib_files;
  for (auto& lib : toConfig->get_lib_files()) {
    lib_files.push_back(lib.c_str());
  }

  timing_engine->set_num_threads(50);
  timing_engine->set_design_work_space(design_work_space);
  timing_engine->readLiberty(lib_files);

  auto idb_adapter = std::make_unique<TimingIDBAdapter>(timing_engine->get_ista());

  idb::IdbBuilder* idb = dmInst->get_idb_builder();
  idb_adapter->set_idb(idb);
  idb_adapter->convertDBToTimingNetlist(true);
  timing_engine->set_db_adapter(std::move(idb_adapter));

  const char* sdc_file = toConfig->get_sdc_file().c_str();
  if (sdc_file != nullptr) {
    timing_engine->readSdc(sdc_file);
  }

  timing_engine->buildGraph();
  timing_engine->updateTiming();

  timingEngine->set_sta_engine(timing_engine);
}

void TimingEngineBuilder::findDrvrVertices()
{
  Netlist* design_nl = timingEngine->get_sta_engine()->get_netlist();
  Net* net;
  FOREACH_NET(design_nl, net)
  {
    DesignObject* driver = net->getDriver();
    if (driver) {
      StaVertex* driver_vertex = timingEngine->get_sta_engine()->findVertex(driver->getFullName().c_str());
      timingEngine->get_driver_vertices().push_back(driver_vertex);
    }
  }

  sort(timingEngine->get_driver_vertices().begin(), timingEngine->get_driver_vertices().end(), [](StaVertex* v1, StaVertex* v2) {
    TOLevel level1 = v1->get_level();
    TOLevel level2 = v2->get_level();
    return (level1 < level2);
  });
}

void TimingEngineBuilder::findBufferCells()
{
  timingEngine->set_buf_lowest_driver_res(nullptr);
  float low_drive = -kInf;

  auto& all_libs = timingEngine->get_sta_engine()->getAllLib();
  for (auto& lib : all_libs) {
    for (auto& cell : lib->get_cells()) {
      if (cell->isBuffer() && timingEngine->canFindLibertyCell(cell.get())) {
        timingEngine->get_buffer_cells().push_back(cell.get());

        LibPort* in_port;
        LibPort* out_port;
        cell->bufferPorts(in_port, out_port);
        float driver_res = out_port->driveResistance();
        if (driver_res > low_drive) {
          low_drive = driver_res;
          timingEngine->set_buf_lowest_driver_res(cell.get());
          cout << cell->get_cell_name() << endl;
        }
      }
    }
  }
  if (timingEngine->get_buffer_cells().empty()) {
    std::cout << "Can't find buffers in liberty file." << std::endl;
    exit(1);
  }
}

void TimingEngineBuilder::calcCellTargetLoads()
{
  // Calc the target slew for all buffers in the libraries.
  calcTargetSlewsForBuffer();
  if (timingEngine->get_target_map() == nullptr) {
    timingEngine->new_target_map();
  } else {
    timingEngine->get_target_map()->clear();
  }

  // Calc target loads
  auto& all_libs = timingEngine->get_sta_engine()->getAllLib();
  int cell_count = 0;
  for (auto& lib : all_libs) {                // lib
    for (auto& libcell : lib->get_cells()) {  // lib cells
      LibCell* cell = libcell.get();

      if (timingEngine->canFindLibertyCell(cell)) {
        cell_count++;
        calcTargetLoad(cell);
      }
    }
  }
}

void TimingEngineBuilder::calcTargetSlewsForBuffer()
{
  timingEngine->get_target_slews() = {0.0};

  TOSlew rise_fall_slew[2]{0.0};  // TransType: kFall / kRise;
  int rise_fall_number[2]{0};
  for (LibCell* buffer : timingEngine->get_buffer_cells()) {
    calcTargetSlewsForBuffer(rise_fall_slew, rise_fall_number, buffer);
  }

  TOSlew rise_arc_slew = rise_fall_slew[TYPE_RISE] / rise_fall_number[TYPE_RISE];
  TOSlew fall_arc_slew = rise_fall_slew[TYPE_FALL] / rise_fall_number[TYPE_FALL];
  timingEngine->get_target_slews()[TYPE_RISE] = max(timingEngine->get_target_slews()[TYPE_RISE], rise_arc_slew);
  timingEngine->get_target_slews()[TYPE_FALL] = max(timingEngine->get_target_slews()[TYPE_FALL], fall_arc_slew);
}

void TimingEngineBuilder::calcTargetSlewsForBuffer(TOSlew rise_fall_slew[], int rise_fall_number[], LibCell* buffer)
{
  LibPort* input;
  LibPort* output;
  buffer->bufferPorts(input, output);

  // get timing arc of (input, output)
  const char* in_name = input->get_port_name();
  const char* out_name = output->get_port_name();

  std::optional<LibArcSet*> arcset = buffer->findLibertyArcSet(in_name, out_name, LibArc::TimingType::kDefault);

  if (arcset.has_value()) {
    auto& arcs = (*arcset)->get_arcs();
    for (auto& arc : arcs) {
      // LibertyTableModel *model = arc->get_table_model();
      calcArcSlew(rise_fall_slew, rise_fall_number, input, TransType::kFall, arc.get());
      calcArcSlew(rise_fall_slew, rise_fall_number, input, TransType::kRise, arc.get());
    }
  }
}

void TimingEngineBuilder::calcArcSlew(TOSlew rise_fall_slew[], int rise_fall_number[], LibPort* port, TransType trans_type, LibArc* arc)
{
  auto cap_value = port->get_port_cap(ista::AnalysisMode::kMaxMin, trans_type);
  float port_cap = cap_value ? *cap_value : port->get_port_cap();
  float cap_load = port_cap * _slew_2_load_cap_factor;

  TOSlew slew1 = arc->getSlewNs(trans_type, 0.01, cap_load);
  TOSlew slew = arc->getSlewNs(trans_type, slew1, cap_load);

  if (trans_type == TransType::kFall) {
    rise_fall_slew[TYPE_FALL] += slew;
    rise_fall_number[TYPE_FALL]++;
  } else {
    rise_fall_slew[TYPE_RISE] += slew;
    rise_fall_number[TYPE_RISE]++;
  }
}

void TimingEngineBuilder::calcTargetLoad(LibCell* cell)
{
  float target_total_load = 0.0;
  int total_arc_num = 0;
  // get cell all arcset
  auto& cell_arcset = cell->get_cell_arcs();
  for (auto& arcset : cell_arcset) {  // arcset
    ieda::Vector<std::unique_ptr<ista::LibArc>>& arcs = arcset->get_arcs();
    for (auto& arc : arcs) {
      if (arc->isDelayArc()
          && !((arc->get_timing_type() == LibArc::TimingType::kNonSeqHoldRising)
               || (arc->get_timing_type() == LibArc::TimingType::kNonSeqHoldFalling)
               || (arc->get_timing_type() == LibArc::TimingType::kNonSeqSetupRising)
               || (arc->get_timing_type() == LibArc::TimingType::kNonSeqSetupFalling))) {
        if ((arc->get_timing_type() == LibArc::TimingType::kComb) || (arc->get_timing_type() == LibArc::TimingType::kCombRise)
            || (arc->get_timing_type() == LibArc::TimingType::kRisingEdge)
            || (arc->get_timing_type() == LibArc::TimingType::kDefault)) {
          calcTargetLoad(target_total_load, total_arc_num, arc.get(), TransType::kRise);
        }

        if ((arc->get_timing_type() == LibArc::TimingType::kComb) || (arc->get_timing_type() == LibArc::TimingType::kCombFall)
            || (arc->get_timing_type() == LibArc::TimingType::kFallingEdge)
            || (arc->get_timing_type() == LibArc::TimingType::kDefault)) {
          calcTargetLoad(target_total_load, total_arc_num, arc.get(), TransType::kFall);
        }
      }
    }
  }

  float target_load = total_arc_num ? target_total_load / total_arc_num : 0.0;
  timingEngine->get_target_map()->insert(cell, target_load);
}

void TimingEngineBuilder::calcTargetLoad(float& target_total_load, int& total_arc_num, LibArc* arc, TransType rf)
{
  float arc_target_load;
  if (arc->isNegativeArc()) {
    if (rf == TransType::kRise) {
      arc_target_load = calcTargetLoad(arc, rf, TransType::kFall);
    } else {
      arc_target_load = calcTargetLoad(arc, rf, rf);
    }
  } else {
    if (rf == TransType::kRise) {
      arc_target_load = calcTargetLoad(arc, rf, rf);
    } else {
      arc_target_load = calcTargetLoad(arc, rf, TransType::kFall);
    }
  }
  target_total_load += arc_target_load;
  total_arc_num++;
}

/**
 * @brief Calc the load capacitance that will result in the output slew matching out_slew.
 *
 * @param cell
 * @param model
 * @param in_type
 * @param out_type
 * @return float
 */
float TimingEngineBuilder::calcTargetLoad(LibArc* arc, TransType in_type, TransType out_type)
{
  if (!arc || !arc->isDelayArc()) {
    return 0.0;
  }

  int    in_rf_index = static_cast<int>(in_type) - 1;
  int    out_rf_index = static_cast<int>(out_type) - 1;
  TOSlew in_slew = timingEngine->get_target_slews()[in_rf_index];
  TOSlew out_slew = timingEngine->get_target_slews()[out_rf_index];

  double low_bound_cap = 0.0;
  double upper_bound_cap = 1.0e-12;

  // 扩展 upper_bound_cap 直到 slew_diff 为正
  while (calcSlewDiffOfGate(in_type, upper_bound_cap, in_slew, out_slew, arc) < 0.0) {
    low_bound_cap = upper_bound_cap;
    upper_bound_cap *= 2.0;
  }

  // 使用二分法找到接近 0 的电容值
  while ((upper_bound_cap - low_bound_cap) / upper_bound_cap > 0.01) {
    double mid_cap = (low_bound_cap + upper_bound_cap) / 2.0;
    double slew_diff_mid = calcSlewDiffOfGate(in_type, mid_cap, in_slew, out_slew, arc);

    if (slew_diff_mid < 0.0) {
      low_bound_cap = mid_cap;
    } else {
      upper_bound_cap = mid_cap;
    }
  }

  return (low_bound_cap + upper_bound_cap) / 2.0;
}

TOSlew TimingEngineBuilder::calcSlewDiffOfGate(TransType in_type, float cap_load, TOSlew in_slew, TOSlew out_slew, LibArc* arc)
{
  TOSlew slew = arc->getSlewNs(in_type, in_slew, cap_load);
  return slew - out_slew;
}

}  // namespace ito