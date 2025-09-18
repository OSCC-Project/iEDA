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
 * @file StaVetex.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of sta vertex.
 * @version 0.1
 * @date 2021-02-10
 */

#include "StaVertex.hh"

#include <ranges>

#include "StaDump.hh"
#include "StaFunc.hh"
#include "delay/ElmoreDelayCalc.hh"
#include "log/Log.hh"

namespace ista {

StaPropagationTag::StaPropagationTag(const StaPropagationTag& orig)
    : _is_false_path(orig._is_false_path),
      _is_min_max_delay(orig._is_min_max_delay),
      _is_multicycle_path(orig._is_multicycle_path) {}

StaPropagationTag& StaPropagationTag::operator=(const StaPropagationTag& orig) {
  if (this != &orig) {
    _is_false_path = orig._is_false_path;
    _is_min_max_delay = orig._is_min_max_delay;
    _is_multicycle_path = orig._is_multicycle_path;
  }
  return *this;
}

/**
 * @brief set prop tag accord tag type.
 *
 * @param tag_type
 */
void StaPropagationTag::setTag(TagType tag_type, bool is_set) {
  switch (tag_type) {
    case TagType::kProp:
      _is_prop = is_set;
      break;

    case TagType::kFalse:
      _is_false_path = is_set;
      break;

    case TagType::kMinMax:
      _is_min_max_delay = is_set;
      break;

    case TagType::kMulticycle:
      _is_multicycle_path = is_set;
      break;

    default:
      LOG_FATAL << "not support.";
      break;
  }
}

/**
 * @brief is Tag set.
 *
 * @param tag_type
 * @return true
 * @return false
 */
bool StaPropagationTag::isTagSet(TagType tag_type) const {
  bool is_set = false;
  switch (tag_type) {
    case TagType::kProp:
      is_set = _is_prop;
      break;

    case TagType::kFalse:
      is_set = _is_false_path;
      break;

    case TagType::kMinMax:
      is_set = _is_min_max_delay;
      break;

    case TagType::kMulticycle:
      is_set = _is_multicycle_path;
      break;

    default:
      LOG_FATAL << "not support.";
      break;
  }
  return is_set;
}

StaVertex::StaVertex(DesignObject* obj)
    : _obj(obj),
      _slew_bucket(c_vertex_slew_data_bucket_size),
      _path_delay_bucket(c_vertex_path_delay_data_bucket_size) {
  if (!obj) {
    DLOG_INFO << "obj is NULL";
  }
}

/**
 * @brief Get check arc(Setup/hold) if the vertex is end node.
 *
 * @param analysis_mode max/min mode.
 * @return StaArc*
 */
StaArc* StaVertex::getSetupHoldArc(AnalysisMode analysis_mode) {
  if (!is_end()) {
    return nullptr;
  }

  FOREACH_SNK_ARC(this, snk_arc) {
    if (IS_MAX(analysis_mode) && snk_arc->isSetupArc()) {
      return snk_arc;
    }

    if (IS_MIN(analysis_mode) && snk_arc->isHoldArc()) {
      return snk_arc;
    }
  }

  return nullptr;
}

/**
 * @brief Get check arc(Setup/hold Recovery/Removal) if the vertex is end node.
 *
 * @param analysis_mode
 * @return StaArc*
 */
StaArc* StaVertex::getCheckArc(AnalysisMode analysis_mode) {
  if (!is_end()) {
    return nullptr;
  }

  FOREACH_SNK_ARC(this, snk_arc) {
    if (IS_MAX(analysis_mode) &&
        (snk_arc->isSetupArc() || snk_arc->isRecoveryArc())) {
      return snk_arc;
    }

    if (IS_MIN(analysis_mode) &&
        (snk_arc->isHoldArc() || snk_arc->isRemovalArc())) {
      return snk_arc;
    }
  }

  return nullptr;
}

/**
 * @brief Get the Src Check Arcs from the vertex.
 *
 * @param src_vertex
 * @return std::vector<StaArc*>
 */
std::vector<StaArc*> StaVertex::getSrcCheckArcs(AnalysisMode analysis_mode) {
  std::vector<StaArc*> ret;
  auto& src_arcs = get_src_arcs();
  for (auto* src_arc :
       src_arcs | std::views::filter([analysis_mode](auto* src_arc) {
         return (IS_MAX(analysis_mode) &&
                 (src_arc->isSetupArc() || src_arc->isRecoveryArc())) ||
                (IS_MIN(analysis_mode) &&
                 (src_arc->isHoldArc() || src_arc->isRemovalArc()));
       })) {
    ret.emplace_back(src_arc);
  }
  return ret;
}

/**
 * @brief Clear src net arcs.
 *
 */
void StaVertex::clearSrcNetArcs() {
  LOG_FATAL_IF(
      !std::erase_if(_src_arcs, [](StaArc* arc) { return arc->isNetArc(); }));
}

/**
 * @brief Clear snk net arcs.
 *
 */
void StaVertex::clearSnkNetArcs() {
  LOG_FATAL_IF(
      !std::erase_if(_snk_arcs, [](StaArc* arc) { return arc->isNetArc(); }));
}

/**
 * @brief Get the owner cell.
 *
 * @return LibCell*
 */
LibCell* StaVertex::getOwnCell() {
  if (auto* pin = dynamic_cast<Pin*>(_obj); pin) {
    return pin->get_own_instance()->get_inst_cell();
  }
  return nullptr;
}

/**
 * @brief Get the owner cell name of the vertex.
 *
 * @return const char*
 */
const char* StaVertex::getOwnCellName() {
  const char* cell_name = (dynamic_cast<Pin*>(_obj))
                              ->get_own_instance()
                              ->get_inst_cell()
                              ->get_cell_name();
  return cell_name;
}

/**
 * @brief Get the owner cell name/port name of the vertex.
 *
 * @return const char*
 */
const char* StaVertex::getOwnCellOrPortName() {
  const char* obj_name;
  if (this->is_port()) {
    obj_name = _obj->get_name();
  } else {
    obj_name = (dynamic_cast<Pin*>(_obj))
                   ->get_own_instance()
                   ->get_inst_cell()
                   ->get_cell_name();
  }

  return obj_name;
}

/**
 * @brief Get name with cell name like u1:A (NAND2X1).
 *
 * @return std::string
 */
std::string StaVertex::getNameWithCellName() {
  if (_obj->isPort()) {
    return getName() + " (port)";
  } else {
    const char* cell_name = getOwnCellName();
    return getName() + Str::printf(" (%s)", cell_name);
  }
}

/**
 * @brief Get the owner inst name/port name of the vertex.
 *
 * @return const char*
 */
const char* StaVertex::getOwnInstanceOrPortName() {
  const char* inst_name;
  if (this->is_port()) {
    inst_name = _obj->get_name();
  } else {
    inst_name = (dynamic_cast<Pin*>(_obj))->get_own_instance()->get_name();
  }

  return inst_name;
}

/**
 * @brief Add slew data.
 *
 * @param slew_data
 */
void StaVertex::addData(StaSlewData* slew_data) {
  if (slew_data) {
    slew_data->set_own_vertex(this);
    _slew_bucket.addData(slew_data, 0);
  }
}

/**
 * @brief Add clock path data.
 *
 * @param data The clock data.
 */
void StaVertex::addData(StaClockData* clock_data) {
  if (clock_data) {
    _clock_bucket.addData(clock_data, 0);
    clock_data->set_own_vertex(this);
  }
}

/**
 * @brief Add data path data.
 *
 * @param delay_data The data path data.
 */
void StaVertex::addData(StaPathDelayData* delay_data) {
  if (delay_data) {
    delay_data->set_own_vertex(this);
    _path_delay_bucket.addData(delay_data, 0);
  }
}

/**
 * @brief Init slew data, if not create zero slew default.
 */
void StaVertex::initSlewData() {
  auto& slew_bucket = getSlewBucket();
  if (!slew_bucket.empty()) {
    return;
  }

  auto construct_slew_data = [](AnalysisMode delay_type, TransType trans_type,
                                StaVertex* own_vertex, int slew) {
    StaSlewData* slew_data =
        new StaSlewData(delay_type, trans_type, own_vertex, slew);
    own_vertex->addData(slew_data);
  };

  /*if not, create default zero slew.*/
  construct_slew_data(AnalysisMode::kMax, TransType::kRise, this, 0);
  construct_slew_data(AnalysisMode::kMax, TransType::kFall, this, 0);
  construct_slew_data(AnalysisMode::kMin, TransType::kRise, this, 0);
  construct_slew_data(AnalysisMode::kMin, TransType::kFall, this, 0);
}

/**
 * @brief Init at data, if not create zero at default.
 * 
 */
void StaVertex::initPathDelayData() {
  auto& data_bucket = getDataBucket();
  if (!data_bucket.empty()) {
    return;
  }

  auto construct_path_delay_data = [](AnalysisMode delay_type, TransType trans_type,
                                StaVertex* own_vertex, int at) {
    auto* path_delay_data =
        new StaPathDelayData(delay_type, trans_type, at, nullptr, own_vertex);
    own_vertex->addData(path_delay_data);
  };

  /*if not, create default zero path delay.*/
  construct_path_delay_data(AnalysisMode::kMax, TransType::kRise, this, 0);
  construct_path_delay_data(AnalysisMode::kMax, TransType::kFall, this, 0);
  construct_path_delay_data(AnalysisMode::kMin, TransType::kRise, this, 0);
  construct_path_delay_data(AnalysisMode::kMin, TransType::kFall, this, 0);

}

/**
 * @brief reset vertex data and arc data for increment analysis.
 *
 */
void StaVertex::resetVertexArcData() {
  if (isResetVertexBucket()) {
    return;
  }

  resetColor();
  // resetLevel();  level for propagation
  reset_is_slew_prop();
  reset_is_delay_prop();
  reset_is_fwd();
  reset_is_bwd();

  resetSlewBucket();
  resetClockBucket();
  resetPathDelayBucket();

  if (is_end()) {
    return;
  }

  FOREACH_SRC_ARC(this, src_arc) {
    if (src_arc->isDelayArc()) {
      if (!(src_arc->isResetArcDelayBucket())) {
        src_arc->resetArcDelayBucket();
      }
      auto* snk_vertex = src_arc->get_snk();
      snk_vertex->resetVertexArcData();
    }
  }

  return;
}

/**
 * @brief Get src arc from the src vertex.
 *
 * @param snk_vertex
 * @return std::vector<StaArc*>
 */
std::vector<StaArc*> StaVertex::getSrcArc(StaVertex* src_vertex) {
  std::vector<StaArc*> ret;
  FOREACH_SNK_ARC(this, snk_arc) {
    if (src_vertex == snk_arc->get_src()) {
      ret.push_back(snk_arc);
    }
  }
  return ret;
}

/**
 * @brief Get snk arc to the snk vertex.
 *
 * @param snk_vertex
 * @return std::vector<StaArc*>
 */
std::vector<StaArc*> StaVertex::getSnkArc(StaVertex* snk_vertex) {
  std::vector<StaArc*> ret;
  FOREACH_SRC_ARC(this, src_arc) {
    if (snk_vertex == src_arc->get_snk()) {
      ret.push_back(src_arc);
    }
  }
  return ret;
}

/**
 * @brief Set the level of the vetex.
 *
 * @param src_level
 * @param snk_level
 */
void StaVertex::set_level(unsigned level) {
  if (_level == 0) {
    _level = level;
  } else {
    if (level > _level) {
      _level = level;
    }
  }
}

/**
 * @brief Get the needed clock data.
 *
 * @param analysis_mode
 * @param trans_type
 * @return std::vector<StaData*>
 */
std::vector<StaData*> StaVertex::getClockData(AnalysisMode analysis_mode,
                                              TransType trans_type) {
  std::vector<StaData*> clock_data_vec;
  StaData* clock_data;
  FOREACH_CLOCK_DATA(this, clock_data) {
    if (clock_data->get_trans_type() == trans_type &&
        clock_data->get_delay_type() == analysis_mode) {
      clock_data_vec.push_back(clock_data);
    }
  }

  return clock_data_vec;
}

/**
 * @brief Get the needed clock.
 *
 * @param analysis_mode
 * @param trans_type
 * @return StaClock*
 */
StaClock* StaVertex::getPropClock(AnalysisMode analysis_mode,
                                  TransType trans_type) {
  StaData* data;
  FOREACH_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* path_delay = dynamic_cast<StaPathDelayData*>(data);
      if (auto prop_clock =
              path_delay->get_launch_clock_data()->get_prop_clock();
          prop_clock) {
        return prop_clock;
      }
    }
  }

  return nullptr;
}

/**
 * @brief get vertex all propagated clocks, for the propagate clock may be more
 * than one clock.
 *
 * @param analysis_mode
 * @param trans_type
 * @param is_data_path
 * @return std::unordered_set<StaClock*>
 */
std::unordered_set<StaClock*> StaVertex::getPropagatedClock(
    AnalysisMode analysis_mode, TransType trans_type, bool is_data_path) {
  std::unordered_set<StaClock*> prop_clocks;
  auto get_data_clock = [&prop_clocks, analysis_mode, trans_type](auto* data) {
    if ((data->get_delay_type() == analysis_mode ||
         AnalysisMode::kMaxMin == analysis_mode) &&
        ((data->get_trans_type() == trans_type) ||
         TransType::kRiseFall == trans_type)) {
      if (data->isPathDelayData()) {
        auto* path_delay = dynamic_cast<StaPathDelayData*>(data);
        if (auto* prop_clock =
                path_delay->get_launch_clock_data()->get_prop_clock();
            prop_clock) {
          if (!prop_clocks.contains(prop_clock)) {
            prop_clocks.insert(prop_clock);
          }
        }
      } else {
        auto* clock_data = dynamic_cast<StaClockData*>(data);
        auto* prop_clock = clock_data->get_prop_clock();
        if (!prop_clocks.contains(prop_clock)) {
          prop_clocks.insert(prop_clock);
        }
      }
    }
  };

  if (is_data_path) {
    StaData* data;
    FOREACH_DELAY_DATA(this, data) { get_data_clock(data); }
  } else {
    StaData* data;
    FOREACH_CLOCK_DATA(this, data) { get_data_clock(data); }
  }

  return prop_clocks;
}

/**
 * @brief Judge the clock name whether prop clock.
 *
 * @param clock_name
 * @return true
 * @return false
 */
bool StaVertex::isPropClock(const char* clock_name, AnalysisMode analysis_mode,
                            TransType trans_type) {
  auto prop_clocks = getPropagatedClock(analysis_mode, trans_type, false);
  auto it = std::find_if(
      prop_clocks.begin(), prop_clocks.end(), [clock_name](auto* prop_clock) {
        return Str::equal(prop_clock->get_clock_name(), clock_name);
      });
  return it != prop_clocks.end();
}

/**
 * @brief Judge the clock pin vertex is rising triggered.
 *
 * @return unsigned
 */
unsigned StaVertex::isRisingTriggered() {
  if (!is_clock()) {
    return 0;
  }

  FOREACH_SRC_ARC(this, src_arc) {
    if (src_arc->isCheckArc() && src_arc->isRisingEdgeCheck()) {
      return 1;
    }
  }

  return 0;
}

/**
 * @brief Judge the clock pin vertex is falling triggered.
 *
 * @return unsigned
 */
unsigned StaVertex::isFallingTriggered() {
  if (!is_clock()) {
    return 0;
  }

  FOREACH_SRC_ARC(this, src_arc) {
    if (src_arc->isCheckArc() && src_arc->isFallingEdgeCheck()) {
      return 1;
    }
  }

  return 0;
}

/**
 * @brief Get arrive time.
 *
 * @param analysis_mode
 * @param trans_type
 * @return int
 */
std::optional<int64_t> StaVertex::getArriveTime(AnalysisMode analysis_mode,
                                                TransType trans_type) {
  StaData* data;
  std::vector<int64_t> arrive_times;
  FOREACH_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* path_delay = dynamic_cast<StaPathDelayData*>(data);
      arrive_times.push_back(path_delay->get_arrive_time());
    }
  }

  if (!arrive_times.empty()) {
    std::sort(arrive_times.begin(), arrive_times.end());
  } else {
    return std::nullopt;
  }

  int64_t arrive_time = (analysis_mode == AnalysisMode::kMax)
                            ? (arrive_times.back())
                            : (arrive_times.front());

  return arrive_time;
}

/**
 * @brief Get arrive time.
 *
 * @param analysis_mode
 * @param trans_type
 * @param clock_name
 * @return int
 */
std::optional<int64_t> StaVertex::getClockArriveTime(
    AnalysisMode analysis_mode, TransType trans_type,
    std::optional<std::string> clock_name) {
  StaData* clock_data;
  std::vector<int64_t> arrive_times;
  FOREACH_CLOCK_DATA(this, clock_data) {
    if (clock_data->get_delay_type() == analysis_mode &&
        clock_data->get_trans_type() == trans_type) {
      if (!clock_name ||
          clock_name.value() == dynamic_cast<StaClockData*>(clock_data)
                                    ->get_prop_clock()
                                    ->get_clock_name()) {
        arrive_times.push_back(
            dynamic_cast<StaClockData*>(clock_data)->get_arrive_time());
      }
    }
  }

  if (!arrive_times.empty()) {
    std::sort(arrive_times.begin(), arrive_times.end());
  } else {
    return std::nullopt;
  }

  int64_t arrive_time = (analysis_mode == AnalysisMode::kMax)
                            ? (arrive_times.back())
                            : (arrive_times.front());

  return arrive_time;
}

/**
 * @brief Get required time.
 *
 * @param analysis_mode
 * @param trans_type
 * @return int
 */
std::optional<int> StaVertex::getReqTime(AnalysisMode analysis_mode,
                                         TransType trans_type) {
  StaData* data;
  std::vector<int> req_times;
  FOREACH_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* path_delay = dynamic_cast<StaPathDelayData*>(data);
      if (auto req_time = path_delay->get_req_time(); req_time) {
        req_times.push_back(*(req_time));
      }
    }
  }

  if (!req_times.empty()) {
    std::sort(req_times.begin(), req_times.end());
  } else {
    return std::nullopt;
  }

  int req_time = (analysis_mode == AnalysisMode::kMax) ? (req_times.front())
                                                       : (req_times.back());

  return req_time;
}

/**
 * @brief Get slack.
 *
 * @param analysis_mode
 * @param trans_type
 * @return int64_t
 */
std::optional<int64_t> StaVertex::getSlack(AnalysisMode analysis_mode,
                                           TransType trans_type) {
  auto arrive_time = getArriveTime(analysis_mode, trans_type);
  auto req_time = getReqTime(analysis_mode, trans_type);
  if (arrive_time && req_time) {
    int64_t slack = (analysis_mode == AnalysisMode::kMax)
                        ? (*req_time - *arrive_time)
                        : (*arrive_time - *req_time);
    return slack;
  }

  return std::nullopt;
}

/**
 * @brief get total negative slack in Ns.
 *
 * @param analysis_mode
 * @return std::optional<double>
 */
std::optional<double> StaVertex::getTNSNs(AnalysisMode analysis_mode) {
  std::optional<double> vertex_tns_ns;
  StaData* data;
  FOREACH_DELAY_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode) {
      auto* path_delay = dynamic_cast<StaPathDelayData*>(data);
      auto at_fs = path_delay->get_arrive_time();
      auto rt_fs = path_delay->get_req_time();
      if (!rt_fs) {
        continue;
      }

      double slack_fs = (analysis_mode == AnalysisMode::kMax)
                            ? (*rt_fs - at_fs)
                            : (at_fs - *rt_fs);
      double slack_ns = FS_TO_NS(slack_fs);
      vertex_tns_ns ? (*vertex_tns_ns) += slack_ns : vertex_tns_ns = slack_ns;
    }
  }

  return vertex_tns_ns;
}

/**
 * @brief Get slew.
 *
 * @param analysis_mode
 * @param trans_type
 * @return std::optional<int>
 */
std::optional<int> StaVertex::getSlew(AnalysisMode analysis_mode,
                                      TransType trans_type) {
  StaData* data;
  FOREACH_SLEW_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* slew_data = dynamic_cast<StaSlewData*>(data);
      return slew_data->get_slew();
    }
  }

  return std::nullopt;
}

/**
 * @brief Find the exist slew data.
 *
 * @param delay_type
 * @param trans_type
 * @param src_slew_data
 * @return StaData*
 */
StaSlewData* StaVertex::getSlewData(AnalysisMode analysis_mode,
                                    TransType trans_type,
                                    StaData* src_slew_data) {
  StaData* data;
  FOREACH_SLEW_DATA(this, data) {
    if ((data->get_delay_type() == analysis_mode) &&
        (data->get_trans_type() == trans_type) &&
        (!src_slew_data || (src_slew_data == data->get_bwd()))) {
      auto* slew_data = dynamic_cast<StaSlewData*>(data);
      return slew_data;
    }
  }

  return nullptr;
}

/**
 * @brief Find the exist path delay data.
 *
 * @param analysis_mode
 * @param trans_type
 * @param src_delay_data
 * @return StaPathDelayData*
 */
StaPathDelayData* StaVertex::getPathDelayData(AnalysisMode analysis_mode,
                                              TransType trans_type,
                                              StaData* src_delay_data) {
  StaData* data;
  FOREACH_DELAY_DATA(this, data) {
    if ((data->get_delay_type() == analysis_mode) &&
        (data->get_trans_type() == trans_type) &&
        (!src_delay_data || (src_delay_data == data->get_bwd()))) {
      auto* delay_data = dynamic_cast<StaPathDelayData*>(data);
      return delay_data;
    }
  }

  return nullptr;
}

/**
 * @brief Get the cap for the driver node, or pin cap for the load node.
 *
 * @param analysis_mode
 * @param trans_type
 * @return double unit is PF.
 */
double StaVertex::getLoad(AnalysisMode analysis_mode, TransType trans_type) {
  double load_or_cap;

  auto* obj = get_design_obj();
  auto* the_net = obj->get_net();
  if (!the_net) {
    return 0.0;
  }

  if (the_net->getDriver() == obj) {
    auto* rc_net = Sta::getOrCreateSta()->getRcNet(the_net);
    load_or_cap = rc_net ? rc_net->load(analysis_mode, trans_type)
                         : the_net->getLoad(analysis_mode, trans_type);
  } else {
    load_or_cap = obj->cap(analysis_mode, trans_type);
  }
  return load_or_cap;
}

/**
 * @brief Get the slew impulse for the net load node.
 * 
 * @param analysis_mode 
 * @param trans_type 
 * @return double 
 */
double StaVertex::getNetSlewImpulse(AnalysisMode analysis_mode, TransType trans_type) {
  double load_impulse = 0.0;

  auto* obj = get_design_obj();
  auto* the_net = obj->get_net();
  if (!the_net) {
    return 0.0;
  }

  if (the_net->getDriver() != obj) {
    auto* rc_net = Sta::getOrCreateSta()->getRcNet(the_net);
    load_impulse = rc_net ? rc_net->slewImpulse(*obj, analysis_mode, trans_type)
                         : 0.0;
  } 

  return load_impulse;

}
/**
 * @brief Get the slew delay for the net load node.
 * 
 * @param analysis_mode 
 * @param trans_type 
 * @return double 
 */
double StaVertex::getNetLoadDelay(AnalysisMode analysis_mode, TransType trans_type) {
  double load_delay = 0.0;

  auto* obj = get_design_obj();
  auto* the_net = obj->get_net();
  if (!the_net) {
    return 0.0;
  }

  if (the_net->getDriver() != obj) {
    auto* rc_net = Sta::getOrCreateSta()->getRcNet(the_net);
    
    if (rc_net) {
      auto* rc_tree = rc_net->rct();
      if (rc_tree) {
        auto* node = rc_tree->node(obj->getFullName());
        load_delay = PS_TO_NS(node->delay(analysis_mode, trans_type));
      }
    }
  } 

  return load_delay;
}

/**
 * @brief get net load include pin cap.
 *
 * @return double
 */
double StaVertex::getNetLoad() {
  auto* obj = get_design_obj();
  auto* the_net = obj->get_net();
  auto* rc_net = Sta::getOrCreateSta()->getRcNet(the_net);
  if (rc_net) {
    double load_include_pin_cap =
        rc_net->load(AnalysisMode::kMax, TransType::kRise);

    return load_include_pin_cap;
  }

  return 0.0;
}

/**
 * @brief Get the resistance for load node of net.
 *
 * @param analysis_mode
 * @param trans_type
 * @return double
 */
double StaVertex::getResistance(AnalysisMode analysis_mode,
                                TransType trans_type) {
  double resistance = 0.0;

  auto* obj = get_design_obj();
  auto* the_net = obj->get_net();
  if (the_net->getDriver() != obj) {
    auto* rc_net = Sta::getOrCreateSta()->getRcNet(the_net);
    resistance =
        rc_net ? rc_net->getResistance(analysis_mode, trans_type, obj) : 0.0;
  }
  return resistance;
}

/**
 * @brief Get network latency.
 *
 * @param analysis_mode
 * @param trans_type
 * @return int
 */
int StaVertex::getNetworkLatency(AnalysisMode analysis_mode,
                                 TransType trans_type) {
  StaData* data;
  FOREACH_CLOCK_DATA(this, data) {
    if (data->get_delay_type() == analysis_mode &&
        data->get_trans_type() == trans_type) {
      auto* latency_data = dynamic_cast<StaClockData*>(data);
      return latency_data->get_arrive_time();
    }
  }

  return 0;
}

/**
 * @brief Get the depth of the vertex in timing path.
 *
 */
void StaVertex::getPathDepth(
    std::priority_queue<int, std::vector<int>, std::greater<int>>&
        depth_min_queue,
    int depth /*= 0*/) {
  depth++;
  if (this->is_start() || this->get_snk_arcs().empty() || this->is_clock()) {
    depth_min_queue.push(depth);
    return;
  }

  FOREACH_SNK_ARC(this, snk_arc) {
    if (snk_arc->isDelayArc()) {
      auto* src_vertex = snk_arc->get_src();
      src_vertex->getPathDepth(depth_min_queue, depth);
    }
  }
}

/**
 * @brief Assume the vertex is endpoint, get the depth of the worst path.
 *
 * @return unsigned
 */
unsigned StaVertex::GetWorstPathDepth(AnalysisMode analysis_mode) {
  auto* ista = Sta::getOrCreateSta();

  auto* rise_worst_seq_data =
      ista->getWorstSeqData(this, analysis_mode, TransType::kRise).front();
  auto* fall_worst_seq_data =
      ista->getWorstSeqData(this, analysis_mode, TransType::kFall).front();

  auto rise_depth = rise_worst_seq_data->getPathDelayData().size();
  auto fall_depth = fall_worst_seq_data->getPathDelayData().size();

  return rise_worst_seq_data->getSlackNs() < fall_worst_seq_data->getSlackNs()
             ? rise_depth
             : fall_depth;
}

/**
 * @brief Execute the sta functor.
 *
 * @param func
 * @return unsigned return 1 if success, or return 0.
 */
unsigned StaVertex::exec(StaFunc& func) { return func(this); }

/**
 * @brief dump vertex info for debug.
 *
 */
void StaVertex::dump() {
  StaDumpYaml dump_data;
  dump_data(this);
  dump_data.printText("vertex.txt");
}

}  // namespace ista
