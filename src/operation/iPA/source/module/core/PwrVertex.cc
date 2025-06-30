// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file PwrVertex.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of the power vertex.
 * @version 0.1
 * @date 2023-01-19
 */
#include "PwrVertex.hh"

#include <algorithm>

#include "PwrArc.hh"
#include "PwrSeqGraph.hh"
#include "ops/dump/PwrDumpGraph.hh"
#include "api/Power.hh"

namespace ipower {

/**
 * @brief sort seq vertex by name.
 *
 * @param lhs
 * @param rhs
 * @return true
 * @return false
 */
bool PwrSeqVertexComp::operator()(const PwrSeqVertex* const& lhs,
                                  const PwrSeqVertex* const& rhs) const {
  return lhs->get_obj_name() > rhs->get_obj_name();
}

/**
 * @brief get drive voltage of the vertex cell.
 *
 * @return std::optional<double>
 */
std::optional<double> PwrVertex::getDriveVoltage() {
  auto* design_obj = _sta_vertex->get_design_obj();
  if (design_obj->isPort()) {
    // for port.
    auto* the_net = design_obj->get_net();
    auto the_loads = the_net->getLoads();
    for (auto* one_load : the_loads) {
      if (one_load->isPin()) {
        design_obj = one_load;
        break;
      }      
    }
  }

  if (design_obj->isPort()) {
    // TODO(to taosimin), fix io volatage.
    return 0.0;
  }

  auto* liberty_port = dynamic_cast<Pin*>(design_obj)->get_cell_port();
  auto* the_library = liberty_port->get_ower_cell()->get_owner_lib();
  double nominal_voltage = the_library->get_nom_voltage();
  return nominal_voltage;
}

/**
 * @brief get the min level of fanout seq vertex.
 *
 * @return std::optional<PwrSeqVertex*>
 */
std::optional<PwrSeqVertex*> PwrVertex::getFanoutMinSeqLevel() {
  std::optional<PwrSeqVertex*> min_seq_vertex;

  std::ranges::for_each(
      _fanout_seq_vertexes, [&min_seq_vertex](auto* seq_vertex) {
        if (!min_seq_vertex) {
          min_seq_vertex = seq_vertex;
        } else {
          if (seq_vertex->get_level() < (*min_seq_vertex)->get_level()) {
            min_seq_vertex = seq_vertex;
          }
        }
      });

  return min_seq_vertex;
}

/**
 * @brief get the vertex own clock domain, maybe more than once when have clock
 * mux.
 *
 * @return std::set<StaClock*>
 */
std::unordered_set<StaClock*> PwrVertex::getOwnClockDomain() {
  auto* the_sta_vertex = get_sta_vertex();
  bool is_data_path = !is_clock_network();
  auto own_clock_domain_set = the_sta_vertex->getPropagatedClock(
      AnalysisMode::kMaxMin, TransType::kRiseFall, is_data_path);
  return own_clock_domain_set;
}

/**
 * @brief get the vertex fastest clock.
 *
 * @return StaClock*
 */
std::optional<StaClock*> PwrVertex::getOwnFastestClockDomain() {
  auto own_clock_domain_set = getOwnClockDomain();
  if (own_clock_domain_set.empty()) {
    return std::nullopt;
  }

  std::optional<StaClock*> the_fastest_clock;
  auto get_fast_clock = [&the_fastest_clock](auto* the_clock) {
    if (!the_fastest_clock) {
      the_fastest_clock = the_clock;
    } else {
      if ((*the_fastest_clock)->getPeriodNs() > the_clock->getPeriodNs()) {
        the_fastest_clock = the_clock;
      }
    }
  };

  std::ranges::for_each(own_clock_domain_set, get_fast_clock);
  return (*the_fastest_clock);
}

/**
 * @brief add toggle data.
 *
 * @param toggle_data
 */
void PwrVertex::addData(PwrToggleData* toggle_data,
                        std::optional<PwrClock*> the_fastest_clock) {
  if (toggle_data) {
    // Set clock domain only for propagation data.
    auto data_source = toggle_data->get_data_source();
    if ((data_source == PwrDataSource::kDataPropagation) ||
        (data_source == PwrDataSource::kClockPropagation)) {
      auto clock_domain = getOwnFastestClockDomain();
      if (clock_domain) {
        toggle_data->set_clock_domain(*clock_domain);
      } else {
        LOG_FATAL_IF(!the_fastest_clock) << "not found fastest clock.";
        toggle_data->set_clock_domain(*the_fastest_clock);
      }
    }
    _toggle_bucket.addData(toggle_data, 0);
  }
}
/**
 * @brief add toggle and sp together.
 *
 * @param toggle
 * @param sp
 * @param data_source
 */
void PwrVertex::addData(double toggle, double sp, PwrDataSource data_source,
                        std::optional<PwrClock*> the_fastest_clock) {
  auto* pwr_toggle_data = new PwrToggleData(data_source, this, toggle);
  auto* pwr_SP_data = new PwrSPData(data_source, this, sp);

  if (the_fastest_clock) {
    addData(pwr_toggle_data, the_fastest_clock);
  } else {
    addData(pwr_toggle_data);
  }
  addData(pwr_SP_data);
}

/**
 * @brief get toggle data from toggle bucket
 *
 * @return double
 */
double PwrVertex::getToggleData(std::optional<PwrDataSource> data_source) {
  // if is a const vertex
  if (is_const()) {
    return 0.0;
  }

  // Get the toggle data of the vertex.
  auto& toggle_bucket = getToggleBucket();

  PwrToggleData* toggle_data = nullptr;
  if (data_source) {
    toggle_data =
        dynamic_cast<PwrToggleData*>(toggle_bucket.frontData(*data_source));
  } else {
    toggle_data = dynamic_cast<PwrToggleData*>(toggle_bucket.frontData());
  }

  if (!toggle_data) {
    Power* ipower = Power::getOrCreatePower(nullptr);
    double default_toggle = ipower->get_default_toggle();
    return default_toggle;
  }
  double toggle_value = toggle_data->get_toggle();
  return toggle_value;
}

/**
 * @brief get the vertex sp data.
 *
 * @return double
 */
double PwrVertex::getSPData(std::optional<PwrDataSource> data_source) {
  // if is a const vertex
  if (is_const()) {
    double sp_value = is_const_gnd() ? 0.0 : 1.0;
    return sp_value;
  }

  // Get the sp data of the vertex.
  auto& sp_bucket = getSPBucket();

  PwrSPData* sp_data = nullptr;
  if (data_source) {
    sp_data = dynamic_cast<PwrSPData*>(sp_bucket.frontData(*data_source));
  } else {
    sp_data = dynamic_cast<PwrSPData*>(sp_bucket.frontData());
  }
  if (!sp_data) {
    return c_default_sp;
  }
  double sp_value = sp_data->get_sp();
  return sp_value;
}

/**
 * @brief judge src vertex have const node.
 *
 * @return true
 * @return false
 */
bool PwrVertex::isHaveConstSrcVertex() {
  bool is_have_const_vertex = false;
  FOREACH_SNK_PWR_ARC(this, the_arc) {
    auto* src_pwr_vertex = the_arc->get_src();
    if (src_pwr_vertex->is_const()) {
      is_have_const_vertex = true;
    }
  }
  return is_have_const_vertex;
}

/**
 * @brief dump vertex info for debug.
 *
 */
void PwrVertex::dumpVertexInfo() {
  PwrDumpGraphYaml dump_yaml;
  dump_yaml(this);
  dump_yaml.printText("vertex_info.txt");
}

}  // namespace ipower
