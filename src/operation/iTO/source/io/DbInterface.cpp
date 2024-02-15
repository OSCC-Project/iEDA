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
    auto  wns = _timing_engine->reportTNS(clk_name, AnalysisMode::kMax);
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
  makeEquivCells();

  findDrvrVertices();
  findBufferCells();
  findCellTargetLoads();
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

void DbInterface::makeEquivCells() {
  vector<LibertyLibrary *> equiv_libs;
  auto &                   all_libs = _timing_engine->getAllLib();
  for (auto &lib : all_libs) {
    for (auto &cell : lib->get_cells()) {
      if (isLinkCell(cell.get())) {
        equiv_libs.push_back(lib.get());
        break;
      }
    }
  }

  _timing_engine->makeEquivCells(equiv_libs);
}

bool DbInterface::isLinkCell(LibertyCell *cell) {
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
    Level level1 = v1->get_level();
    Level level2 = v2->get_level();
    return (level1 < level2);
  });
}

void DbInterface::findBufferCells() {
  _lowest_drive_buffer = nullptr;
  float low_drive = -kInf;

  auto &all_libs = _timing_engine->getAllLib();
  for (auto &lib : all_libs) {
    for (auto &cell : lib->get_cells()) {
      if (cell->isBuffer() && isLinkCell(cell.get())) {
        _buffer_cells.push_back(cell.get());

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
  if (_buffer_cells.empty()) {
    std::cout << "No buffers found." << std::endl;
    exit(1);
  }
}

void DbInterface::findBufferTargetSlews() {
  _target_slews = {0.0};

  Slew slews[2]{0.0}; // TransType: kFall / kRise;
  int  counts[2]{0};
  for (LibertyCell *buffer : _buffer_cells) {
    findBufferTargetSlews(buffer, slews, counts);
  }

  Slew slew_rise = slews[_rise] / counts[_rise];
  Slew slew_fall = slews[_fall] / counts[_fall];
  if (slew_rise > _target_slews[_rise]) {
    _target_slews[_rise] = slew_rise;
    _target_slews[_fall] = slew_fall;
  }
}

void DbInterface::findBufferTargetSlews(LibertyCell *buffer,
                                        // Return values.
                                        Slew slews[], int counts[]) {
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
                              // Return values.
                              Slew slews[], int counts[]) {
  auto  cap_value = port->get_port_cap(ista::AnalysisMode::kMaxMin, trans_type);
  float in_cap = cap_value ? *cap_value : port->get_port_cap();
  float load_cap = in_cap * _tgt_slew_load_cap_factor;

  Slew slew1 = arc->getSlewNs(trans_type, 0.01, load_cap);
  Slew slew = arc->getSlewNs(trans_type, slew1, load_cap);

  if (trans_type == TransType::kFall) {
    slews[_fall] += slew;
    counts[_fall]++;
  } else {
    slews[_rise] += slew;
    counts[_rise]++;
  }
}

///////////////////////////////////////////////////////////////////////////////////////

void DbInterface::findCellTargetLoads() {
  // Find target slew across all buffers in the libraries.
  findBufferTargetSlews();
  if (_cell_target_load_map == nullptr) {
    _cell_target_load_map = new CellTargetLoadMap;
  }
  _cell_target_load_map->clear();

  // Find target loads
  auto &all_libs = _timing_engine->getAllLib();
  int   cell_count = 0;
  for (auto &lib : all_libs) {               // lib
    for (auto &libcell : lib->get_cells()) { // lib cells
      LibertyCell *cell = libcell.get();

      if (isLinkCell(cell)) {
        cell_count++;
        findTargetLoad(cell);
      }
    }
  }
}

void DbInterface::findTargetLoad(LibertyCell *cell) {
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
          findTargetLoad(cell, arc.get(), TransType::kRise, target_load_sum, arc_count);
        }

        if ((arc->get_timing_type() == LibertyArc::TimingType::kComb) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kCombFall) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kFallingEdge) ||
            (arc->get_timing_type() == LibertyArc::TimingType::kDefault)) {
          findTargetLoad(cell, arc.get(), TransType::kFall, target_load_sum, arc_count);
        }
      }
    }
  }

  float target_load = arc_count ? target_load_sum / arc_count : 0.0;
  (*_cell_target_load_map)[cell] = target_load;
}

void DbInterface::findTargetLoad(LibertyCell *cell, LibertyArc *arc, TransType rf,
                                 // return values
                                 float &target_load_sum, int &arc_count) {
  float arc_target_load;
  if (arc->isNegativeArc()) {
    if (rf == TransType::kRise) {
      arc_target_load = findTargetLoad(cell, arc, rf, TransType::kFall);
    } else {
      arc_target_load = findTargetLoad(cell, arc, rf, rf);
    }
  } else {
    if (rf == TransType::kRise) {
      arc_target_load = findTargetLoad(cell, arc, rf, rf);
    } else {
      arc_target_load = findTargetLoad(cell, arc, rf, TransType::kFall);
    }
  }
  target_load_sum += arc_target_load;
  arc_count++;
}

/**
 * @brief Find the load capacitance that will cause the output slew to be equal
 * to out_slew.
 *
 * @param cell
 * @param model
 * @param in_type
 * @param out_type
 * @return float
 */
float DbInterface::findTargetLoad(LibertyCell *cell, LibertyArc *arc, TransType in_type,
                                  TransType out_type) {
  if (arc && arc->isDelayArc()) {
    int  in_rf_index = (int)in_type - 1;
    int  out_rf_index = (int)out_type - 1;
    Slew in_slew = _target_slews[in_rf_index];
    Slew out_slew = _target_slews[out_rf_index];

    double load_cap1 = 0.0;
    double load_cap2 = 1.0e-12; // 1pF
    double tol = .01;           // 1%

    double diff1 = gateSlewDiff(in_type, cell, load_cap1, in_slew, out_slew, arc);
    if (diff1 > 0.0) {
      // Zero load cap out_slew is higher than the target.
      return 0.0;
    }
    double diff2 = gateSlewDiff(in_type, cell, load_cap2, in_slew, out_slew, arc);
    // binary search for diff = 0.
    while (abs(load_cap1 - load_cap2) > max(load_cap1, load_cap2) * tol) {
      if (diff2 < 0.0) {
        load_cap1 = load_cap2;
        load_cap2 *= 2;
        diff2 = gateSlewDiff(in_type, cell, load_cap2, in_slew, out_slew, arc);
      } else {
        double load_cap3 = (load_cap1 + load_cap2) / 2.0;
        double diff3 = gateSlewDiff(in_type, cell, load_cap3, in_slew, out_slew, arc);
        if (diff3 < 0.0) {
          load_cap1 = load_cap3;
        } else {
          load_cap2 = load_cap3;
          diff2 = diff3;
        }
      }
    } // end while
    return load_cap1;
  }
  return 0.0;
}

Slew DbInterface::gateSlewDiff(TransType in_type, LibertyCell *cell, float load_cap,
                               Slew in_slew, Slew out_slew, LibertyArc *arc) {
  Slew slew = arc->getSlewNs(in_type, in_slew, load_cap);
  return slew - out_slew;
}

bool DbInterface::overMaxArea() {
  double max_utilization = _config->get_max_utilization();
  // initBlock();
  double core_area = DesignCalculator::calculateCoreArea(_core, _dbu);
  double max_area_utilize = core_area * max_utilization;
  return fuzzyGreaterEqual(_design_area, max_area_utilize);
}

} // namespace ito
