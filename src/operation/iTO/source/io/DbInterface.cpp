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
#include "DbInterface.h"
#include "DesignCalculator.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "builder.h"

namespace ito {
DbInterface *DbInterface::_db_interface = nullptr;
int          DbInterface::_rise = (int)TransType::kRise - 1;
int          DbInterface::_fall = (int)TransType::kFall - 1;

DbInterface *DbInterface::get_db_interface(ToConfig *config, IdbBuilder *idb,
                                           TimingEngine *timing) {
  static std::mutex mt;
  if (_db_interface == nullptr) {
    std::lock_guard<std::mutex> lock(mt);
    if (_db_interface == nullptr) {
      _db_interface = new DbInterface(config);
      _db_interface->_timing_engine = timing;
      _db_interface->_idb = idb;
      _db_interface->initData();
    }
  }
  return _db_interface;
}

void DbInterface::destroyDbInterface() {
  if (_db_interface != nullptr) {
    delete _db_interface;
    _db_interface = nullptr;
  }
}

void DbInterface::set_eval_data() {
  if (!_eval_data.empty()) {
    _eval_data.clear();
  }
  auto clk_list = _timing_engine->getClockList();
  for (auto clk : clk_list) {
    auto clk_name = clk->get_clock_name();
    auto  wns = _timing_engine->reportWNS(clk_name, AnalysisMode::kMax);
    auto  tns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
    auto  freq = 1000.0 / (clk->getPeriodNs() - wns);
    _eval_data.push_back({clk_name, wns, tns, freq});
  }
}

void DbInterface::initData() {
  // log report
  string report_path = _config->get_report_file();
  _reporter = new Reporter(report_path);
  _reporter->reportTime(true);
  _reporter->get_ofstream() << _config->get_def_file() << endl;
  _reporter->get_ofstream().close();

  // placer
  _placer = new Placer(_idb);

  initDbData();
  findEquivLibCells();

  findDrvrVertices();
  findBufferCells();
  calcCellTargetLoads();
}

void DbInterface::initDbData() {
  IdbLefService *idb_lef_service = _idb->get_lef_service();
  IdbLayout *    idb_layout = idb_lef_service->get_layout();

  IdbCore *idb_core = idb_layout->get_core();
  IdbRect *idb_rect = idb_core->get_bounding_box();

  _dbu = idb_layout->get_units()->get_micron_dbu();
  _core = Rectangle(idb_rect->get_low_x(), idb_rect->get_high_x(), idb_rect->get_low_y(),
                    idb_rect->get_high_y());

  IdbDefService *idb_def_service = _idb->get_def_service();
  IdbDesign *    idb_design = idb_def_service->get_design();
  _layout = new ito::Layout(idb_design);
  _design_area = DesignCalculator::calculateDesignArea(_layout, _dbu);
}

void DbInterface::findEquivLibCells() {
  vector<LibertyLibrary *> equiv_libs;
  auto &                   all_libs = _timing_engine->getAllLib();
  for (auto &lib : all_libs) {
    for (auto &cell : lib->get_cells()) {
      if (canFindLibertyCell(cell.get())) {
        equiv_libs.push_back(lib.get());
        break;
      }
    }
  }

  _timing_engine->makeEquivCells(equiv_libs);
}

bool DbInterface::canFindLibertyCell(LibertyCell *cell) {
  return _timing_engine->findLibertyCell(cell->get_cell_name()) == cell;
}

void DbInterface::findDrvrVertices() {
  Netlist *design_nl = _timing_engine->get_netlist();
  Net *    net;
  FOREACH_NET(design_nl, net) {
    DesignObject *driver = net->getDriver();
    if (driver) {
      StaVertex *drvr_vertex = _timing_engine->findVertex(driver->getFullName().c_str());
      _drvr_vertices.push_back(drvr_vertex);
    }
  }

  sort(_drvr_vertices.begin(), _drvr_vertices.end(), [](StaVertex *v1, StaVertex *v2) {
    TOLevel level1 = v1->get_level();
    TOLevel level2 = v2->get_level();
    return (level1 < level2);
  });
}

void DbInterface::findBufferCells() {
  _lowest_drive_buffer = nullptr;
  float low_drive = -kInf;

  auto &all_libs = _timing_engine->getAllLib();
  for (auto &lib : all_libs) {
    for (auto &cell : lib->get_cells()) {
      if (cell->isBuffer() && canFindLibertyCell(cell.get())) {
        _available_buffer_cells.push_back(cell.get());

        LibertyPort *in_port;
        LibertyPort *out_port;
        cell->bufferPorts(in_port, out_port);
        float drvr_res = out_port->driveResistance();
        if (drvr_res > low_drive) {
          low_drive = drvr_res;
          _lowest_drive_buffer = cell.get();
        }
      }
    }
  }
  if (_available_buffer_cells.empty()) {
    std::cout << "Can't find buffers in liberty file." << std::endl;
    exit(1);
  }
}

void DbInterface::calcTargetSlewsForBuffer() {
  _target_slews = {0.0};

  TOSlew slews[2]{0.0}; // TransType: kFall / kRise;
  int  counts[2]{0};
  for (LibertyCell *buffer : _available_buffer_cells) {
    calcTargetSlewsForBuffer(buffer, slews, counts);
  }

  TOSlew slew_rise = slews[_rise] / counts[_rise];
  TOSlew slew_fall = slews[_fall] / counts[_fall];
  if (slew_rise > _target_slews[_rise]) {
    _target_slews[_rise] = slew_rise;
    _target_slews[_fall] = slew_fall;
  }
}

void DbInterface::calcTargetSlewsForBuffer(LibertyCell *buffer,
                                           TOSlew slews[], int counts[]) {
  LibertyPort *input;
  LibertyPort *output;
  buffer->bufferPorts(input, output);

  // get timing arc of (input, output)
  const char *in_name = input->get_port_name();
  const char *out_name = output->get_port_name();

  std::optional<LibertyArcSet *> arcset =
      buffer->findLibertyArcSet(in_name, out_name, LibertyArc::TimingType::kDefault);

  if (arcset.has_value()) {
    auto &arcs = (*arcset)->get_arcs();
    for (auto &arc : arcs) {
      // LibertyTableModel *model = arc->get_table_model();
      getGateSlew(input, TransType::kFall, arc.get(), slews, counts);
      getGateSlew(input, TransType::kRise, arc.get(), slews, counts);
    }
  }
}

void DbInterface::getGateSlew(LibertyPort *port, TransType trans_type, LibertyArc *arc,
                              TOSlew slews[], int counts[]) {
  auto  cap_value = port->get_port_cap(ista::AnalysisMode::kMaxMin, trans_type);
  float in_cap = cap_value ? *cap_value : port->get_port_cap();
  float load_cap = in_cap * _slew_2_load_cap_factor;

  TOSlew slew1 = arc->getSlewNs(trans_type, 0.01, load_cap);
  TOSlew slew = arc->getSlewNs(trans_type, slew1, load_cap);

  if (trans_type == TransType::kFall) {
    slews[_fall] += slew;
    counts[_fall]++;
  } else {
    slews[_rise] += slew;
    counts[_rise]++;
  }
}

///////////////////////////////////////////////////////////////////////////////////////

void DbInterface::calcCellTargetLoads() {
  // Calc the target slew for all buffers in the libraries.
  calcTargetSlewsForBuffer();
  if (_cell_target_load_map == nullptr) {
    _cell_target_load_map = new TOLibCellLoadMap;
  }
  _cell_target_load_map->clear();

  // Calc target loads
  auto &all_libs = _timing_engine->getAllLib();
  int   cell_count = 0;
  for (auto &lib : all_libs) {               // lib
    for (auto &libcell : lib->get_cells()) { // lib cells
      LibertyCell *cell = libcell.get();

      if (canFindLibertyCell(cell)) {
        cell_count++;
        calcTargetLoad(cell);
      }
    }
  }
}

void DbInterface::calcTargetLoad(LibertyCell *cell) {
  float target_load_sum = 0.0;
  int   arc_count = 0;
  // get cell all arcset
  auto &cell_arcset = cell->get_cell_arcs();
  for (auto &arcset : cell_arcset) { // arcset
    ieda::Vector<std::unique_ptr<ista::LibertyArc>> &arcs = arcset->get_arcs();
    for (auto &arc : arcs) {
      if (arc->isDelayArc() &&
          !((arc->get_timing_type() == LibertyArc::TimingType::kNonSeqHoldRising) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kNonSeqHoldFalling) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kNonSeqSetupRising) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kNonSeqSetupFalling))) {
        if ((arc->get_timing_type() == LibertyArc::TimingType::kComb) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kCombRise) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kRisingEdge) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kDefault)) {
          calcTargetLoad(arc.get(), TransType::kRise, target_load_sum, arc_count);
        }

        if ((arc->get_timing_type() == LibertyArc::TimingType::kComb) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kCombFall) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kFallingEdge) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kDefault)) {
          calcTargetLoad(arc.get(), TransType::kFall, target_load_sum, arc_count);
        }
      }
    }
  }

  float target_load = arc_count ? target_load_sum / arc_count : 0.0;
  (*_cell_target_load_map)[cell] = target_load;
}

void DbInterface::calcTargetLoad(LibertyArc *arc, TransType rf,
                                 float &target_load_sum, int &arc_count) {
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
  target_load_sum += arc_target_load;
  arc_count++;
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
float DbInterface::calcTargetLoad(LibertyArc *arc, TransType in_type,
                                  TransType out_type) {
  if (arc && arc->isDelayArc()) {
    int    in_rf_index = (int)in_type - 1;
    int    out_rf_index = (int)out_type - 1;
    TOSlew in_slew = _target_slews[in_rf_index];
    TOSlew out_slew = _target_slews[out_rf_index];

    double low_bound_cap = 0.0;
    double upper_bound_cap = 1.0e-12;
    double tolerate = 0.01; // 1%

    double slew_diff_1 =
        calcSlewDiffOfGate(in_type, low_bound_cap, in_slew, out_slew, arc);
    if (slew_diff_1 > 0.0) {
      return 0.0;
    }
    double slew_diff_2 =
        calcSlewDiffOfGate(in_type, upper_bound_cap, in_slew, out_slew, arc);
    // calc diff = 0 by binary search.
    while (abs(low_bound_cap - upper_bound_cap) >
           max(low_bound_cap, upper_bound_cap) * tolerate) {
      if (slew_diff_2 < 0.0) {
        low_bound_cap = upper_bound_cap;
        upper_bound_cap *= 2;
        slew_diff_2 =
            calcSlewDiffOfGate(in_type, upper_bound_cap, in_slew, out_slew, arc);
      } else {
        double load_cap3 = (low_bound_cap + upper_bound_cap) / 2.0;
        double slew_diff_3 =
            calcSlewDiffOfGate(in_type, load_cap3, in_slew, out_slew, arc);
        if (slew_diff_3 < 0.0) {
          low_bound_cap = load_cap3;
        } else {
          upper_bound_cap = load_cap3;
          slew_diff_2 = slew_diff_3;
        }
      }
    } // end while
    return low_bound_cap;
  }
  return 0.0;
}

TOSlew DbInterface::calcSlewDiffOfGate(TransType in_type, float load_cap, TOSlew in_slew,
                                       TOSlew out_slew, LibertyArc *arc) {
  TOSlew slew = arc->getSlewNs(in_type, in_slew, load_cap);
  return slew - out_slew;
}

bool DbInterface::overMaxArea() {
  double max_utilization = _config->get_max_utilization();
  // initBlock();
  double core_area = DesignCalculator::calculateCoreArea(_core, _dbu);
  double max_area_utilize = core_area * max_utilization;
  return approximatelyGreaterEqual(_design_area, max_area_utilize);
}

} // namespace ito
