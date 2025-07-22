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
 * @file Lib.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief This is the implemention of liberty module.
 * @version 0.1
 * @date 2020-11-28
 */

#include "Lib.hh"

#include <fstream>
#include <functional>
#include <set>
#include <utility>

#include "json/json.hpp"
#include "solver/Interpolation.hh"
#include "string/StrMap.hh"

namespace ista {

LibAxis::LibAxis(const char* axis_name) : _axis_name(axis_name)
{
}

LibAxis::LibAxis(LibAxis&& other) noexcept : _axis_name(other._axis_name), _axis_values(std::move(other._axis_values))
{
  other._axis_name = nullptr;
}

LibAxis& LibAxis::operator=(LibAxis&& rhs) noexcept
{
  if (this != &rhs) {
    _axis_name = rhs._axis_name;
    rhs._axis_name = nullptr;

    _axis_values = std::move(rhs._axis_values);
  }

  return *this;
}

double LibAxis::operator[](std::size_t index)
{
  return _axis_values[index]->getFloatValue();
}

const std::map<std::string, LibTable::TableType> LibTable::_str2TableType = {
    {"cell_rise", TableType::kCellRise},
    {"cell_fall", TableType::kCellFall},
    {"rise_transition", TableType::kRiseTransition},
    {"fall_transition", TableType::kFallTransition},
    {"rise_constraint", TableType::kRiseConstrain},
    {"fall_constraint", TableType::kFallConstrain},
    {"output_current_rise", TableType::kRiseCurrent},
    {"output_current_fall", TableType::kFallCurrent},
    {"rise_power", TableType::kRisePower},
    {"fall_power", TableType::kFallPower},
    {"ocv_sigma_cell_rise", TableType::kCellRiseSigma},
    {"ocv_sigma_cell_fall", TableType::kCellFallSigma},
    {"ocv_sigma_rise_transition", TableType::kRiseTransitionSigma},
    {"ocv_sigma_fall_transition", TableType::kFallTransitionSigma},
};

LibTable::LibTable(TableType table_type, LibLutTableTemplate* table_template) : _table_type(table_type), _table_template(table_template)
{
}

LibTable::LibTable(LibTable&& other) noexcept
    : _axes(std::move(other._axes)), _table_values(std::move(other._table_values)), _table_type(other._table_type)
{
}

LibTable& LibTable::operator=(LibTable&& rhs) noexcept
{
  if (this != &rhs) {
    _axes = std::move(rhs._axes);
    _table_values = std::move(rhs._table_values);
    _table_type = rhs._table_type;
  }

  return *this;
}

/**
 * @Brief : get axes or template axes.
 * @return auto&
 */
Vector<std::unique_ptr<LibAxis>>& LibTable::get_axes()
{
  if (_axes.empty()) {
    auto* table_template = get_table_template();
    auto& template_table_axes = table_template->get_axes();
    return template_table_axes;
  }
  return _axes;
}

/**
 * @Brief : get axes according index.
 * @param  index
 * @return LibAxis&
 */
LibAxis& LibTable::getAxis(unsigned int index)
{
  return *(get_axes()[index]);
}

/**
 * @brief Lookup the table to find the delay or slew value.
 *
 * @param slew
 * @param constrain_slew_or_load
 * @return double The delay or slew value.
 */
double LibTable::findValue(double slew, double constrain_slew_or_load)
{
  auto* table_template = get_table_template();
  if (!table_template) {
    // fix scalar template is null.
    return get_table_values()[0]->getFloatValue();
  }

  double val1;
  double val2;
  switch (*(table_template->get_template_variable1())) {
    case LibLutTableTemplate::Variable::INPUT_NET_TRANSITION:
    case LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION:
    // power
    case LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME:
      if (auto variable2 = table_template->get_template_variable2(); variable2) {
        LOG_FATAL_IF(*variable2 != LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE
                     && *variable2 != LibLutTableTemplate::Variable::CONSTRAINED_PIN_TRANSITION);
      }

      val1 = slew;
      val2 = constrain_slew_or_load;
      break;

    case LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE:
    case LibLutTableTemplate::Variable::CONSTRAINED_PIN_TRANSITION:
      if (auto variable2 = table_template->get_template_variable2(); variable2) {
        LOG_FATAL_IF(*variable2 != LibLutTableTemplate::Variable::INPUT_NET_TRANSITION
                     && *variable2 != LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION
                     && *variable2 != LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME);
      }

      val1 = constrain_slew_or_load;
      val2 = slew;
      break;

    default:
      LOG_FATAL << "lut table " << get_file_name() << " " << get_line_no() << " invalid delay lut template variable";
      break;
  }

  // first check that slew and constrain_slew_or_load are within the table
  // ranges 
  auto check_val = [this](auto axis_index, auto val) {
    auto num_val = getAxis(axis_index).get_axis_size();
    auto min_val = getAxis(axis_index)[0];
    auto max_val = getAxis(axis_index)[num_val - 1];

    if ((val < min_val) || (val > max_val)) {
      LOG_ERROR_FIRST_N(10) << "Warning: val outside table ranges:  "
                            << "val = " << val << "; min_val = " << min_val << "; max_val = " << max_val << std::endl;
    }
    return num_val;
  };

  // Find the interpolation interval on the axis, 
  // and return the two endpoint values required for interpolation and the left index value.
  auto get_axis_region = [this](auto axis_index, auto num_val, auto val) {
    auto x2 = 0.0;
    unsigned int val_index = 0;
    for (; val_index < num_val; val_index++) {
      x2 = getAxis(axis_index)[val_index];
      if (x2 > val) {
        break;
      }
    }

    if (val_index == num_val) {
      val_index = num_val - 2;
    } else if (val_index) {
      --val_index;
    } else {
      x2 = getAxis(axis_index)[1];
    }
    auto x1 = getAxis(axis_index)[val_index];

    return std::make_tuple(x1, x2, val_index);
  };

  auto get_table_value = [this](auto index) {
    auto& table_values = get_table_values();
    LOG_FATAL_IF(index >= table_values.size()) << "index " << index << " beyond table value size " << table_values.size();
    return table_values[index]->getFloatValue();
  };

  if (1 == get_axes().size()) {
    // Use linear interpolation (LinearInterpolate) in the case of a single variable
    auto num_val1 = check_val(0, val1);
    auto [x1, x2, val1_index] = get_axis_region(0, num_val1, val1);
    unsigned int x1_table_val = get_table_value(val1_index);
    unsigned int x2_table_val = get_table_value(val1_index + 1);

    auto result = LinearInterpolate(x1, x2, x1_table_val, x2_table_val, val1);
    return result;

  } else {
    // Use bilinear interpolation in the case of two variables (BilinearInterpolation)
    auto num_val1 = check_val(0, val1);
    auto num_val2 = check_val(1, val2);

    auto [x1, x2, val1_index] = get_axis_region(0, num_val1, val1);
    auto [y1, y2, val2_index] = get_axis_region(1, num_val2, val2);

    // now do the table lookup
    unsigned int index = num_val2 * val1_index + val2_index;
    const auto q11 = get_table_value(index);

    index = num_val2 * (val1_index + 1) + val2_index;
    const auto q21 = get_table_value(index);

    index = num_val2 * val1_index + (val2_index + 1);
    const auto q12 = get_table_value(index);

    index = num_val2 * (val1_index + 1) + (val2_index + 1);
    const auto q22 = get_table_value(index);

    auto result = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, val1, val2);

    // LOG_ERROR_IF_EVERY_N(result < 0.0, 100) << "table " << get_file_name() <<
    // " " << get_line_no() << " "
    //                                         << "delay value less zero.";
    return result;
  }
}

/**
 * @brief Use slew/Cload for the highest Cload, which approximates output
 * admittance as the "drive".
 *
 * @return double
 */
double LibTable::driveResistance()
{
  double out_cap = 1.0;
  double out_slew = 0.0;

  auto* table_template = get_table_template();
  if (table_template) {
    auto var1 = table_template->get_template_variable1();
    auto var2 = table_template->get_template_variable2();
    if (var1 && *var1 == LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE) {
      auto num_val1 = getAxis(0).get_axis_size();
      out_cap = getAxis(0)[num_val1 - 1];
      out_slew = findValue(0.0, out_cap);
    } else if (var2 && *var2 == LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE) {
      auto num_val2 = getAxis(1).get_axis_size();
      out_cap = getAxis(1)[num_val2 - 1];
      out_slew = findValue(0.0, out_cap);
    }
    // Clip negative slews to zero.
    if (out_slew < 0.0) {
      out_slew = 0.0;
    }
  }

  return out_slew / out_cap;
}

LibVectorTable::LibVectorTable(TableType table_type, LibLutTableTemplate* table_template) : LibTable(table_type, table_template)
{
}

LibVectorTable::LibVectorTable(LibVectorTable&& other) noexcept : LibTable(std::move(other)), _ref_time(other._ref_time)
{
}

LibVectorTable& LibVectorTable::operator=(LibVectorTable&& rhs) noexcept
{
  if (this != &rhs) {
    LibTable::operator=(std::move(rhs));
    _ref_time = rhs._ref_time;
  }
  return *this;
}

/**
 * @brief Get the current vector simulation total time.
 *
 * @return double
 */
std::tuple<double, int> LibVectorTable::getSimulationTotalTimeAndNumPoints()
{
  LibAxis& time_axis = getAxis(_time_index);
  auto num_points = time_axis.get_axis_size();
  double last_simultaion_time_point = time_axis[num_points - 1];
  return {last_simultaion_time_point - _ref_time, num_points};
}

/**
 * @brief Get the output current from ref time, interval time each time until
 * the end simulation time.
 *
 * @param interval
 * @return std::vector<double>
 */
std::vector<double> LibVectorTable::getOutputCurrent(std::optional<LibCurrentSimuInfo>& simu_info)
{
  LibAxis& time_axis = getAxis(_time_index);
  auto& time_axis_value = time_axis.get_axis_values();
  auto& table_values = get_table_values();

  double ref_time = _ref_time;
  if (!simu_info) {
    simu_info = LibCurrentSimuInfo{0, time_axis[time_axis.get_axis_size() - 1] - ref_time, static_cast<int>(time_axis.get_axis_size())};
  }

  auto get_time_index = [&time_axis_value](double current_time, int start_index) -> int {
    int axis_size = time_axis_value.size();
    while (start_index < axis_size) {
      double time_value = time_axis_value[start_index]->getFloatValue();
      if ((time_value > current_time) || IsDoubleEqual(time_value, current_time, 0.000000001)) {
        break;
      }
      ++start_index;
    }
    LOG_FATAL_IF(start_index >= axis_size) << "start index beyond axis size.";
    return start_index;
  };

  auto get_time_and_current = [&time_axis_value, &table_values](int index) {
    return std::make_tuple(time_axis_value[index]->getFloatValue(), std::abs(table_values[index]->getFloatValue()));
  };

  std::vector<double> output_currents;
  double start_time = ref_time + simu_info->_start_time;
  double end_time = ref_time + simu_info->_end_time;
  double interval = (simu_info->_end_time - simu_info->_start_time) / (simu_info->_num_sim_point - 1);

  int start_index = 0;
  for (double current_time = start_time; (current_time < end_time) || IsDoubleEqual(current_time, end_time, 0.000000001);
       current_time += interval) {
    start_index = get_time_index(current_time, start_index);
    int time_index = start_index == 0 ? 1 : start_index;
    auto [upper_time, upper_current] = get_time_and_current(time_index--);
    auto [lower_time, lower_current] = get_time_and_current(time_index);
    double output_current = LinearInterpolate(lower_time, upper_time, lower_current, upper_current, current_time);
    output_currents.push_back(output_current);
  }

  LOG_FATAL_IF(simu_info->_num_sim_point != output_currents.size()) << "output currents size is not equal sim point num.";

  return output_currents;
}

LibCurrentData::LibCurrentData(LibVectorTable* low_low, LibVectorTable* low_high, LibVectorTable* high_low, LibVectorTable* high_high,
                               double slew, double load)
    : _low_low(low_low), _low_high(low_high), _high_low(high_low), _high_high(high_high), _slew(slew), _load(load)
{
}

/**
 * @brief Get the current of simulation total time.
 *
 * @return std::tuple<double, int> total time and point
 */
std::tuple<double, int> LibCurrentData::getSimulationTotalTimeAndNumPoints()
{
  BTreeMap<double, int> total_simulation_times;

  for (auto* table : {_low_low, _low_high, _high_low, _high_high}) {
    auto [total_time, num_point] = table->getSimulationTotalTimeAndNumPoints();
    total_simulation_times[total_time] = num_point;
  }

  auto it = std::min_element(std::begin(total_simulation_times), std::end(total_simulation_times));

  return {it->first, it->second};
}

/**
 * @brief Get the output current use interpolation method.
 *
 */
std::vector<double> LibCurrentData::getOutputCurrent(std::optional<LibCurrentSimuInfo>& simu_info)
{
  auto get_output_currents = [&, this]() {
    return std::make_tuple(_low_low->getOutputCurrent(simu_info), _low_high->getOutputCurrent(simu_info),
                           _high_low->getOutputCurrent(simu_info), _high_high->getOutputCurrent(simu_info));
  };

  auto get_slew_cap = [](LibVectorTable* current_table) {
    // assume the first dimension is slew, the second dimension is load,fixme
    double slew = current_table->getAxis(0)[0];
    double load = current_table->getAxis(1)[0];

    return std::make_tuple(slew, load);
  };

  auto [low_low_current, low_high_current, high_low_current, high_high_current] = get_output_currents();

  auto min_size = std::min({low_low_current.size(), low_high_current.size(), high_low_current.size(), high_high_current.size()});
  auto [low_slew, low_load] = get_slew_cap(_low_low);
  auto [high_slew, high_load] = get_slew_cap(_high_high);

  std::vector<double> output_currents;
  for (size_t index = 0; index < min_size; ++index) {
    if (IsDoubleEqual(low_slew, high_slew)) {
      output_currents.push_back(LinearInterpolate(low_load, high_load, low_low_current[index], low_high_current[index], _load));
    } else if (IsDoubleEqual(low_load, high_load)) {
      output_currents.push_back(LinearInterpolate(low_slew, high_slew, low_low_current[index], high_low_current[index], _slew));
    } else {
      auto output_current = BilinearInterpolation(low_low_current[index], low_high_current[index], high_low_current[index],
                                                  high_high_current[index], low_slew, high_slew, low_load, high_load, _slew, _load);
      output_currents.push_back(output_current);
    }
  }

  return output_currents;
}

LibCCSTable::LibCCSTable(LibTable::TableType table_type) : _table_type(table_type)
{
}

LibDelayTableModel::LibDelayTableModel(LibDelayTableModel&& other) noexcept : _tables(std::move(other._tables))
{
}

LibDelayTableModel& LibDelayTableModel::operator=(LibDelayTableModel&& rhs) noexcept
{
  if (this != &rhs) {
    _tables = std::move(rhs._tables);
  }

  return *this;
}

/**
 * @brief Get the gate delay of the cell arc.
 *
 * @param trans_type Rise/Fall.
 * @param slew The slew.
 * @param load The constrain_slew_or_load.
 * @return double The delay.
 */
std::optional<double> LibDelayTableModel::gateDelay(TransType trans_type, double slew, double load)
{
  LibTable* table = nullptr;
  if (trans_type == TransType::kRise) {
    table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kCellRise));
  } else {
    table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kCellFall));
  }

  if (!table) {
    return std::nullopt;
  }

  return table->findValue(slew, load);
}

/**
 * @brief Get the gate delay sigma of the cell arc.
 *
 * @param mode The early or late.
 * @param trans_type Rise/Fall.
 * @param slew The slew.
 * @param load The constrain_slew_or_load.
 * @return double The delay.
 */
std::optional<double> LibDelayTableModel::gateDelaySigma(AnalysisMode mode, TransType trans_type, double slew, double load)
{
  int index = CAST_TYPE_TO_INDEX(trans_type);
  int shift_index = calcShiftIndex(LibTable::CornerType::kEarly);
  if (mode == AnalysisMode::kMax) {
    shift_index = calcShiftIndex(LibTable::CornerType::kLate);
  }
  LibTable* table = getTable(index + shift_index);

  if (nullptr == table) {
    return std::nullopt;
  }

  return table->findValue(slew, load);
}

/**
 * @brief Get the gate slew of the cell arc.
 *
 * @param trans_type Rise/Fall.
 * @param slew The slew.
 * @param load The constrain_slew_or_load.
 * @return std::optional<double> The slew.
 */
std::optional<double> LibDelayTableModel::gateSlew(TransType trans_type, double slew, double load)
{
  LibTable* table = nullptr;
  if (trans_type == TransType::kRise) {
    table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kRiseTransition));
  } else {
    table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kFallTransition));
  }

  if (!table) {
    return std::nullopt;
  }

  return table->findValue(slew, load);
}

/**
 * @brief Get the gate slew sigma of the cell arc.
 *
 * @param mode The early or late.
 * @param trans_type Rise/Fall.
 * @param slew The slew.
 * @param load The constrain_slew_or_load.
 * @return std::optional<double> The slew.
 */
std::optional<double> LibDelayTableModel::gateSlewSigma(AnalysisMode mode, TransType trans_type, double slew, double load)
{
  int index = CAST_TYPE_TO_INDEX(trans_type);
  int shift_index = calcShiftIndex(LibTable::CornerType::kEarly);
  if (mode == AnalysisMode::kMax) {
    shift_index = calcShiftIndex(LibTable::CornerType::kLate);
  }
  LibTable* table = getTable(index + shift_index);

  if (nullptr == table) {
    return std::nullopt;
  }

  return table->findValue(slew, load);
}

/**
 * @brief Get the gate output current of the cell output.
 *
 * @param trans_type
 * @param slew
 * @param load
 * @return double
 */
std::unique_ptr<LibCurrentData> LibDelayTableModel::gateOutputCurrent(TransType trans_type, double slew, double load)
{
  int table_index;
  if (trans_type == TransType::kRise) {
    table_index = CAST_CURRENT_TYPE_TO_INDEX(LibTable::TableType::kRiseCurrent);
  } else {
    table_index = CAST_CURRENT_TYPE_TO_INDEX(LibTable::TableType::kFallCurrent);
  }

  auto* current_table = _current_tables.at(table_index).get();

  if (!current_table) {
    return nullptr;
  }

  using axis_info = std::pair<double, double>;

  std::map<axis_info, int> axis_info_to_index;
  std::set<double> slew_axis;
  std::set<double> load_axis;

  auto& vector_tables = current_table->get_vector_tables();
  for (int i = 0; auto& vector_table : vector_tables) {
    auto& axes = vector_table->get_axes();

    double axis_slew = axes[0]->get_axis_values().front()->getFloatValue();
    double axis_load = axes[1]->get_axis_values().front()->getFloatValue();

    axis_info_to_index.emplace(std::piecewise_construct, std::forward_as_tuple(axis_slew, axis_load), std::forward_as_tuple(i));
    slew_axis.insert(axis_slew);
    load_axis.insert(axis_load);

    i++;
  }

  auto find_bound = [](std::set<double>& axis_data, double the_data) {
    auto p = axis_data.begin();
    auto q = axis_data.begin();
    for (; p != axis_data.end(); q = p++) {
      if (*p > the_data) {
        break;
      }
    }

    if (p == q) {
      if (p == axis_data.begin()) {
        ++p;
      }
    }
    return std::make_tuple(*q, *p);
  };

  auto [slew_low, slew_high] = find_bound(slew_axis, slew);
  auto [load_low, load_high] = find_bound(load_axis, load);

  auto get_vector_table = [&axis_info_to_index, &vector_tables](double slew, double load) {
    int index = axis_info_to_index[{slew, load}];
    return vector_tables[index].get();
  };

  auto* low_low = get_vector_table(slew_low, load_low);
  auto* low_high = get_vector_table(slew_low, load_high);
  auto* high_low = get_vector_table(slew_high, load_low);
  auto* high_high = get_vector_table(slew_high, load_high);

  auto current_data = std::make_unique<LibCurrentData>(low_low, low_high, high_low, high_high, slew, load);

  return current_data;
}

/**
 * @brief The output driver resistance estimate.
 *
 * @return double
 */
double LibDelayTableModel::driveResistance()
{
  double rise_resistance = 0.0;
  double fall_resistance = 0.0;

  LibTable* rise_table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kRiseTransition));
  if (rise_table) {
    rise_resistance = rise_table->driveResistance();
  }

  LibTable* fall_table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kFallTransition));
  if (fall_table) {
    fall_resistance = fall_table->driveResistance();
  }

  return (rise_resistance > fall_resistance) ? rise_resistance : fall_resistance;
}

/**
 * @brief Get the gate check constrain of the cell arc.
 *
 * @param trans_type Rise/Fall.
 * @param slew The slew.
 * @param constrain_slew The constrain_slew.
 * @return double The slew.
 */
std::optional<double> LibCheckTableModel::gateCheckConstrain(TransType trans_type, double slew, double constrain_slew)
{
  LibTable* table = nullptr;
  if (trans_type == TransType::kRise) {
    table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kRiseConstrain));
  } else {
    table = getTable(CAST_TYPE_TO_INDEX(LibTable::TableType::kFallConstrain));
  }

  if (!table) {
    return std::nullopt;
  }

  return table->findValue(slew, constrain_slew);
}

LibCheckTableModel::LibCheckTableModel(LibCheckTableModel&& other) noexcept : _tables(std::move(other._tables))
{
}

LibCheckTableModel& LibCheckTableModel::operator=(LibCheckTableModel&& rhs) noexcept
{
  if (this != &rhs) {
    _tables = std::move(rhs._tables);
  }

  return *this;
}

LibPowerTableModel::LibPowerTableModel(LibPowerTableModel&& other) noexcept : _tables(std::move(other._tables))
{
}

LibPowerTableModel& LibPowerTableModel::operator=(LibPowerTableModel&& rhs) noexcept
{
  if (this != &rhs) {
    _tables = std::move(rhs._tables);
  }

  return *this;
}

/**
 * @brief Get the gate power of the cell arc.
 *
 * @param trans_type Rise/Fall.
 * @param slew The slew.
 * @param load The constrain_slew_or_load.
 * @return double The power.
 */
double LibPowerTableModel::gatePower(TransType trans_type, double slew, std::optional<double> load)
{
  LibTable* table = nullptr;
  if (trans_type == TransType::kRise) {
    table = getTable(CAST_POWER_TYPE_TO_INDEX(LibTable::TableType::kRisePower));
  } else {
    table = getTable(CAST_POWER_TYPE_TO_INDEX(LibTable::TableType::kFallPower));
  }

  return table->findValue(slew, load.value_or(0.0));
}

LibPort::LibPort(const char* port_name) : _port_name(port_name)
{
}

LibPort::LibPort(LibPort&& other) noexcept
    : _port_name(std::move(other._port_name)), _ower_cell(other._ower_cell), _port_type(other._port_type)
{
}

LibPort& LibPort::operator=(LibPort&& rhs) noexcept
{
  if (this != &rhs) {
    _port_name = std::move(rhs._port_name);
    _ower_cell = rhs._ower_cell;
    _port_type = rhs._port_type;
  }

  return *this;
}

/**
 * @brief Set cap of max/min, rise/fall.
 *
 * @param mode
 * @param trans_type
 * @param cap
 */
void LibPort::set_port_cap(AnalysisMode mode, TransType trans_type, double cap)
{
  if (IS_MAX(mode)) {
    if (IS_RISE(trans_type)) {
      _port_caps[static_cast<int>(LibertyCapIndex::kMaxRise)] = cap;
    } else {
      _port_caps[static_cast<int>(LibertyCapIndex::kMaxFall)] = cap;
    }
  }

  if (IS_MIN(mode)) {
    if (IS_RISE(trans_type)) {
      _port_caps[static_cast<int>(LibertyCapIndex::kMinRise)] = cap;
    } else {
      _port_caps[static_cast<int>(LibertyCapIndex::kMinFall)] = cap;
    }
  }
}

/**
 * @brief Get port cap accord max/min, rise/fall.
 *
 * @param mode
 * @param trans_type
 * @return double
 */
std::optional<double> LibPort::get_port_cap(AnalysisMode mode, TransType trans_type)
{
  if (IS_MAX(mode)) {
    if (IS_RISE(trans_type)) {
      return _port_caps[static_cast<int>(LibertyCapIndex::kMaxRise)];
    } else {
      return _port_caps[static_cast<int>(LibertyCapIndex::kMaxFall)];
    }
  } else {
    if (IS_RISE(trans_type)) {
      return _port_caps[static_cast<int>(LibertyCapIndex::kMinRise)];
    } else {
      return _port_caps[static_cast<int>(LibertyCapIndex::kMinFall)];
    }
  }
}

/**
 * @brief Set cap limit of max/min.
 *
 * @param mode
 * @param cap
 */
void LibPort::set_port_cap_limit(AnalysisMode mode, double cap_limit)
{
  if (IS_MAX(mode)) {
    _cap_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMax)] = cap_limit;
  }

  if (IS_MIN(mode)) {
    _cap_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMin)] = cap_limit;
  }
}

/**
 * @brief Get port cap limit accord max/min.
 *
 * @param mode
 * @return std::optional<double>
 */
std::optional<double> LibPort::get_port_cap_limit(AnalysisMode mode)
{
  if (IS_MAX(mode)) {
    return _cap_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMax)];
  } else {
    return _cap_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMin)];
  }
}

/**
 * @brief Set cap limit of max/min.
 *
 * @param mode
 * @param cap
 */
void LibPort::set_port_slew_limit(AnalysisMode mode, double slew_limit)
{
  if (IS_MAX(mode)) {
    _slew_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMax)] = slew_limit;
  }

  if (IS_MIN(mode)) {
    _slew_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMin)] = slew_limit;
  }
}

/**
 * @brief Get port cap limit accord max/min.
 *
 * @param mode
 * @return std::optional<double>
 */
std::optional<double> LibPort::get_port_slew_limit(AnalysisMode mode)
{
  if (IS_MAX(mode)) {
    return _slew_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMax)];
  } else {
    return _slew_limits[static_cast<int>(LibertyMaxMinLimitIndex::kMin)];
  }
}

/**
 * @brief Calc port drive resistance use slew/cap.
 *
 * @return double
 */
double LibPort::driveResistance()
{
  auto to_arc_sets = _ower_cell->findLibertyArcSet(get_port_name());
  double max_resistance = 0.0;
  for (auto* to_arc_set : to_arc_sets) {
    auto& to_arcs = to_arc_set->get_arcs();
    for (auto& to_arc : to_arcs) {
      double resistance = to_arc->getDriveResistance();
      if (max_resistance < resistance) {
        max_resistance = resistance;
      }
    }
  }
  return max_resistance;
}

/**
 * @brief judge whether liberty port is clock.
 *
 * @return true
 * @return false
      }
    }
  }
  return false;
}

/**
 * @brief judge wherther liberty port is data in according whether has
 * clear/preset arc.
 *
 * @return true
 * @return false
 */
bool LibPort::isSeqDataIn()
{
  auto* liberty_cell = get_ower_cell();
  for (auto& liberty_arc_set : liberty_cell->get_cell_arcs()) {
    auto& lib_arc = liberty_arc_set->get_arcs().front();
    if (lib_arc->isClearPresetArc()) {
      if (Str::equal(lib_arc->get_src_port(), get_port_name())) {
        return false;
      }
    }
  }
  return true;
}

LibPortBus::LibPortBus(const char* port_bus_name) : LibPort(port_bus_name)
{
}

LibLeakagePower::LibLeakagePower() : _owner_cell(nullptr)
{
}
LibLeakagePower::LibLeakagePower(LibLeakagePower&& other) noexcept
    : _related_pg_port(std::move(other._related_pg_port)),
      _when(std::move(other._when)),
      _value(std::move(other._value)),
      _owner_cell(other._owner_cell)
{
}

LibLeakagePower& LibLeakagePower::operator=(LibLeakagePower&& rhs) noexcept
{
  if (this != &rhs) {
    _related_pg_port = std::move(rhs._related_pg_port);
    _when = std::move(rhs._when);
    _value = std::move(rhs._value);
    _owner_cell = rhs._owner_cell;

    rhs._related_pg_port = nullptr;
    rhs._when = nullptr;
    rhs._value = 0;
  }

  return *this;
}

BTreeMap<std::string, LibArc::TimingType> LibArc::_str_to_type = {{"setup_rising", TimingType::kSetupRising},
                                                                  {"hold_rising", TimingType::kHoldRising},
                                                                  {"recovery_rising", TimingType::kRecoveryRising},
                                                                  {"removal_rising", TimingType::kRemovalRising},
                                                                  {"rising_edge", TimingType::kRisingEdge},
                                                                  {"preset", TimingType::kPreset},
                                                                  {"clear", TimingType::kClear},
                                                                  {"three_state_enable", TimingType::kThreeStateEnable},
                                                                  {"three_state_enable_rise", TimingType::kThreeStateEnableRise},
                                                                  {"three_state_enable_fall", TimingType::kThreeStateEnableFall},
                                                                  {"three_state_disable", TimingType::kThreeStateDisable},
                                                                  {"three_state_disable_rise", TimingType::kThreeStateDisableRise},
                                                                  {"three_state_disable_fall", TimingType::kThreeStateDisableFall},
                                                                  {"setup_falling", TimingType::kSetupFalling},
                                                                  {"hold_falling", TimingType::kHoldFalling},
                                                                  {"recovery_falling", TimingType::kRecoveryFalling},
                                                                  {"removal_falling", TimingType::kRemovalFalling},
                                                                  {"falling_edge", TimingType::kFallingEdge},
                                                                  {"min_pulse_width", TimingType::kMinPulseWidth},
                                                                  {"combinational", TimingType::kComb},
                                                                  {"combinational_rise", TimingType::kCombRise},
                                                                  {"combinational_fall", TimingType::kCombFall},
                                                                  {"non_seq_setup_rising", TimingType::kNonSeqSetupRising},
                                                                  {"non_seq_setup_falling", TimingType::kNonSeqSetupFalling},
                                                                  {"non_seq_hold_falling", TimingType::kNonSeqHoldFalling},
                                                                  {"non_seq_hold_rising", TimingType::kNonSeqHoldRising},
                                                                  {"non_seq_hold_falling", TimingType::kNonSeqHoldFalling},
                                                                  {"skew_rising", TimingType::kSkewRising},
                                                                  {"skew_falling", TimingType::kSkewFalling},
                                                                  {"minimum_period", TimingType::kMinimunPeriod},
                                                                  {"max_clock_tree_path", TimingType::kMaxClockTree},
                                                                  {"min_clock_tree_path", TimingType::kMinClockTree},
                                                                  {"nochange_high_high", TimingType::kNoChangeHighHigh},
                                                                  {"nochange_high_low", TimingType::kNoChangeHighLow},
                                                                  {"nochange_low_high", TimingType::kNoChangeLowHigh},
                                                                  {"nochange_low_low", TimingType::kNoChangeLowLow}};

LibArc::LibArc() : _owner_cell(nullptr), _timing_sense(TimingSense::kDefault), _timing_type(TimingType::kDefault)
{
}

LibArc::LibArc(LibArc&& other) noexcept
    : _src_port(std::move(other._src_port)),
      _snk_port(std::move(other._snk_port)),
      _owner_cell(other._owner_cell),
      _timing_sense(other._timing_sense),
      _timing_type(other._timing_type),
      _table_model(std::move(other._table_model))
{
  other._table_model = nullptr;
}

LibArc& LibArc::operator=(LibArc&& rhs) noexcept
{
  if (this != &rhs) {
    _src_port = std::move(rhs._src_port);
    _snk_port = std::move(rhs._snk_port);
    _owner_cell = rhs._owner_cell;
    _timing_sense = rhs._timing_sense;
    _timing_type = rhs._timing_type;
    _table_model = std::move(rhs._table_model);

    rhs._src_port = nullptr;
    rhs._snk_port = nullptr;
    rhs._table_model = nullptr;
  }

  return *this;
}

/**
 * @brief Set arc timing sense.
 *
 * @param timing_sense
 */
void LibArc::set_timing_sense(const char* timing_sense)
{
  if (Str::equal(timing_sense, "positive_unate")) {
    _timing_sense = TimingSense::kPositiveUnate;
  } else if (Str::equal(timing_sense, "negative_unate")) {
    _timing_sense = TimingSense::kNegativeUnate;
  } else {
    _timing_sense = TimingSense::kNonUnate;
  }
}

/**
 * @brief Set arc timing type.
 *
 * @param timing_type
 */
void LibArc::set_timing_type(const char* timing_type)
{
  std::string timing_type_str = timing_type;
  if (_str_to_type.contains(timing_type)) {
    _timing_type = _str_to_type[timing_type_str];
  }
}

/**
 * @brief check the trans type match the arc type.
 *
 * @param trans_type
 * @return true
 * @return false
 */
bool LibArc::isMatchTimingType(TransType trans_type)
{
  bool is_match = true;
  auto timing_type = get_timing_type();
  if ((timing_type == TimingType::kCombFall) && (trans_type == TransType::kRise)) {
    is_match = false;
  } else if ((timing_type == TimingType::kCombRise) && (trans_type == TransType::kFall)) {
    is_match = false;
  } else {
    // TODO, need add more timing type.
  }

  return is_match;
}

/**
 * @brief Judge arc type whether is check arc.
 *
 * @return unsigned 1 if true, else 0.
 */
unsigned LibArc::isCheckArc()
{
  static std::set<TimingType> check_types{TimingType::kSetupRising,     TimingType::kHoldRising,    TimingType::kRecoveryRising,
                                          TimingType::kRemovalRising,   TimingType::kSetupFalling,  TimingType::kHoldFalling,
                                          TimingType::kRecoveryFalling, TimingType::kRemovalFalling};
  auto search = check_types.find(_timing_type);
  return search != check_types.end();
}

/**
 * @brief Judge arc type whether is delay arc.
 *
 * @return unsigned 1 if true, else 0.
 */
unsigned LibArc::isDelayArc()
{
  static std::set<TimingType> delay_types{TimingType::kRisingEdge, TimingType::kFallingEdge, TimingType::kCombRise,
                                          TimingType::kCombFall,   TimingType::kComb,        TimingType::kDefault};
  auto search = delay_types.find(_timing_type);
  return search != delay_types.end();
}

/**
 * @brief Judge arc type whether is min pulse width arc.
 *
 * @return unsigned 1 if true, else 0.
 */
unsigned LibArc::isMpwArc()
{
  return _timing_type == TimingType::kMinPulseWidth;
}

/**
 * @brief judge the liberty arc is clock gate arc.
 *
 * @return unsigned
 */
unsigned LibArc::isClockGateCheckArc()
{
  const char* src_port_name = this->get_src_port();
  const char* snk_port_name = this->get_snk_port();
  auto* src_port = _owner_cell->get_cell_port_or_port_bus(src_port_name);
  LOG_FATAL_IF(!src_port) << "src port " << src_port_name << " is not found.";
  auto* snk_port = _owner_cell->get_cell_port_or_port_bus(snk_port_name);
  LOG_FATAL_IF(!snk_port) << "snk port " << snk_port_name << " is not found.";

  return (_owner_cell->get_is_clock_gating_integrated_cell() && src_port->get_clock_gate_clock_pin()
          && snk_port->get_clock_gate_enable_pin());
}
/**
 * @brief Get the arc delay or constrain value.
 *
 * @param trans_type The transtion type, rise/fall.
 * @param slew The first axis value.
 * @param index2 The second axis value.
 * @return double The delay or constrain value in ns.
 */
double LibArc::getDelayOrConstrainCheckNs(TransType trans_type, double slew, double load_or_constrain_slew)
{
  // get/set time unit of liberty and derate
  TimeUnit input_time_unit = TimeUnit::kNS;
  TimeUnit liberty_time_unit = get_owner_cell()->get_owner_lib()->get_time_unit();
  double input_to_liberty_convert = 1.0;
  double liberty_to_output_convert = 1.0;

  // set convert derate of units
  if (TimeUnit::kPS == liberty_time_unit) {
    input_to_liberty_convert = 1e3;
    liberty_to_output_convert = 1e-3;
  } else if (TimeUnit::kFS == liberty_time_unit) {
    input_to_liberty_convert = 1e6;
    liberty_to_output_convert = 1e-6;
  }

  // pass converted slew into `gateDelay()` and return conveted Delay
  std::optional<double> found_delay;
  if (isDelayArc()) {
    found_delay = _table_model->gateDelay(trans_type, slew * input_to_liberty_convert, load_or_constrain_slew);
  } else {
    found_delay = _table_model->gateCheckConstrain(trans_type, slew * input_to_liberty_convert, load_or_constrain_slew);
  }

  if (found_delay) {
    double ret_value = (*found_delay) * liberty_to_output_convert;
    return ret_value;
  }

  return 0.0;
}

/**
 * @brief Get the arc delay sigma or constrain value.
 *
 * @param mode The late or early.
 * @param trans_type The transtion type, rise/fall.
 * @param slew The first axis value.
 * @param index2 The second axis value.
 * @return double The delay or constrain value in ns.
 */
double LibArc::getDelaySigma(AnalysisMode mode, TransType trans_type, double slew, double load_or_constrain_slew)
{
  // get/set time unit of liberty and derate
  TimeUnit input_time_unit = TimeUnit::kNS;
  TimeUnit liberty_time_unit = get_owner_cell()->get_owner_lib()->get_time_unit();
  double input_to_liberty_convert = 1.0;
  double liberty_to_output_convert = 1.0;

  // set convert derate of units
  if (TimeUnit::kPS == liberty_time_unit) {
    input_to_liberty_convert = 1e3;
    liberty_to_output_convert = 1e-3;
  } else if (TimeUnit::kFS == liberty_time_unit) {
    input_to_liberty_convert = 1e6;
    liberty_to_output_convert = 1e-6;
  }

  // pass converted slew into `gateDelay()` and return conveted Delay
  /*
  std::optional<double> found_delay;
  if (isDelayArc()) {
    found_delay = _table_model->gateDelay(trans_type, slew *
  input_to_liberty_convert, load_or_constrain_slew); } else { found_delay =
  _table_model->gateCheckConstrain(trans_type, slew * input_to_liberty_convert,
  load_or_constrain_slew);
  }
  */
  auto found_delay = _table_model->gateDelaySigma(mode, trans_type, slew * input_to_liberty_convert, load_or_constrain_slew);

  if (found_delay) {
    double ret_value = (*found_delay) * liberty_to_output_convert;
    return ret_value;
  }

  return 0.0;
}

/**
 *
 * @param trans_type The transtion type, rise/fall.
 * @param slew The first axis value.
 * @param load The second axis value.
 * @return double The slew value in ns.
 */
double LibArc::getSlewNs(TransType trans_type, double slew, double load)
{
  if (!isDelayArc()) {
    LOG_FATAL << "check arc has not output slew.";
  }

  // set/get time units in liberty
  TimeUnit input_time_unit = TimeUnit::kNS;
  TimeUnit liberty_time_unit = get_owner_cell()->get_owner_lib()->get_time_unit();
  double input_to_liberty_convert = 1.0;
  double liberty_to_output_convert = 1.0;

  double slew_derate_from_library = get_owner_cell()->get_owner_lib()->get_slew_derate_from_library();

  // set convert derate
  if (TimeUnit::kPS == liberty_time_unit) {
    input_to_liberty_convert = 1e3;
    liberty_to_output_convert = 1e-3;
  } else if (TimeUnit::kFS == liberty_time_unit) {
    input_to_liberty_convert = 1e6;
    liberty_to_output_convert = 1e-6;
  }

  // pass converted slew into `gateSlew()` and return in ns
  auto found_slew = _table_model->gateSlew(trans_type, slew * input_to_liberty_convert, load);

  if (found_slew) {
    double ret_value = (*found_slew) * liberty_to_output_convert * slew_derate_from_library;
    return ret_value;
  }

  return 0.0;
}
/**
 *
 * @param mode The early or late.
 * @param trans_type The transtion type, rise/fall.
 * @param slew The first axis value.
 * @param load The second axis value.
 * @return double The slew value in ns.
 */
double LibArc::getSlewSigma(AnalysisMode mode, TransType trans_type, double slew, double load)
{
  if (!isDelayArc()) {
    LOG_FATAL << "check arc has not output slew.";
  }

  // set/get time units in liberty
  TimeUnit input_time_unit = TimeUnit::kNS;
  TimeUnit liberty_time_unit = get_owner_cell()->get_owner_lib()->get_time_unit();
  double input_to_liberty_convert = 1.0;
  double liberty_to_output_convert = 1.0;

  double slew_derate_from_library = get_owner_cell()->get_owner_lib()->get_slew_derate_from_library();

  // set convert derate
  if (TimeUnit::kPS == liberty_time_unit) {
    input_to_liberty_convert = 1e3;
    liberty_to_output_convert = 1e-3;
  } else if (TimeUnit::kFS == liberty_time_unit) {
    input_to_liberty_convert = 1e6;
    liberty_to_output_convert = 1e-6;
  }

  // pass converted slew into `gateSlew()` and return in ns
  auto found_slew = _table_model->gateSlewSigma(mode, trans_type, slew * input_to_liberty_convert, load);

  if (found_slew) {
    double ret_value = (*found_slew) * liberty_to_output_convert * slew_derate_from_library;
    return ret_value;
  }

  return 0.0;
}

/**
 * @brief Get the arc output current.
 *
 * @param trans_type The transtion type, rise/fall.
 * @param slew The first axis value.
 * @param load The second axis value.
 * @return std::unique_ptr<LibCurrentData> The current values.
 */
std::unique_ptr<LibCurrentData> LibArc::getOutputCurrent(TransType trans_type, double slew, double load)
{
  if (!isDelayArc()) {
    LOG_FATAL << "check arc has not output current.";
  }
  auto current_data = _table_model->gateOutputCurrent(trans_type, slew, load);
  return current_data;
}

LibArcSet::LibArcSet(LibArcSet&& other) noexcept : _arcs(std::move(other._arcs))
{
}

LibArcSet& LibArcSet::operator=(LibArcSet&& rhs) noexcept
{
  if (this != &rhs) {
    _arcs = std::move(rhs._arcs);
  }
  return *this;
}

LibPowerArc::LibPowerArc() : _owner_cell(nullptr)
{
}

LibPowerArc::LibPowerArc(LibPowerArc&& other) noexcept
    : _src_port(std::move(other._src_port)),
      _snk_port(std::move(other._snk_port)),
      _owner_cell(other._owner_cell),
      _internal_power_info(std::move(other._internal_power_info))
{
}

LibPowerArc& LibPowerArc::operator=(LibPowerArc&& rhs) noexcept
{
  if (this != &rhs) {
    _src_port = std::move(rhs._src_port);
    _snk_port = std::move(rhs._snk_port);
    _owner_cell = rhs._owner_cell;
    _internal_power_info = std::move(rhs._internal_power_info);

    rhs._src_port = nullptr;
    rhs._snk_port = nullptr;
  }

  return *this;
}

LibPowerArcSet::LibPowerArcSet(LibPowerArcSet&& other) noexcept : _power_arcs(std::move(other._power_arcs))
{
}

LibPowerArcSet& LibPowerArcSet::operator=(LibPowerArcSet&& rhs) noexcept
{
  if (this != &rhs) {
    _power_arcs = std::move(rhs._power_arcs);
  }
  return *this;
}

LibCell::LibCell(const char* cell_name, LibLibrary* owner_lib)
    : _cell_name(cell_name), _owner_lib(owner_lib), _is_dont_use(0), _is_macro_cell(0)
{
}

LibCell::~LibCell()
{
}

LibCell::LibCell(LibCell&& other) noexcept
    : _cell_name(std::move(other._cell_name)),
      _cell_ports(std::move(other._cell_ports)),
      _cell_arcs(std::move(other._cell_arcs)),
      _cell_power_arcs(std::move(other._cell_power_arcs))
{
}

LibCell& LibCell::operator=(LibCell&& rhs) noexcept
{
  if (this != &rhs) {
    _cell_name = std::move(rhs._cell_name);
    _cell_ports = std::move(rhs._cell_ports);
    _cell_arcs = std::move(rhs._cell_arcs);
    _cell_power_arcs = std::move(rhs._cell_power_arcs);
  }

  return *this;
}

std::vector<LibLeakagePower*> LibCell::getLeakagePowerList()
{
  std::vector<LibLeakagePower*> leakage_power_list;
  for (auto& leakage_power : _leakage_power_list) {
    leakage_power_list.push_back(leakage_power.get());
  }
  return leakage_power_list;
}

void LibCell::addLibertyArc(std::unique_ptr<LibArc>&& cell_arc)
{
  auto arc_set = findLibertyArcSet(cell_arc->get_src_port(), cell_arc->get_snk_port(), cell_arc->get_timing_type());

  if (arc_set) {
    (*arc_set)->addLibertyArc(std::move(cell_arc));
  } else {
    auto* new_arc_set = new LibArcSet();
    _cell_arcs.emplace_back(new_arc_set);
    new_arc_set->addLibertyArc(std::move(cell_arc));
  }
}

void LibCell::addLibertyPowerArc(std::unique_ptr<LibPowerArc>&& cell_power_arc)
{
  auto power_arc_set = findLibertyPowerArcSet(cell_power_arc->get_src_port(), cell_power_arc->get_snk_port());

  if (power_arc_set) {
    (*power_arc_set)->addLibertyPowerArc(std::move(cell_power_arc));
  } else {
    auto* new_power_arc_set = new LibPowerArcSet();
    _cell_power_arcs.emplace_back(new_power_arc_set);
    new_power_arc_set->addLibertyPowerArc(std::move(cell_power_arc));
  }
}

/**
 * @brief Get cell port or port bus.
 *
 * @param port_name
 * @return LibObject* The port or the port bus.
 */
LibPort* LibCell::get_cell_port_or_port_bus(const char* port_name)
{
  // find the port.
  if (auto p = _str2ports.find(port_name); p != _str2ports.end()) {
    return p->second;
  }

  // find the port bus.
  auto [bus_name, index] = Str::matchBusName(port_name);

  if (auto p = _str2portbuses.find(bus_name.c_str()); p != _str2portbuses.end()) {
    if (!index) {
      return p->second;
    } else {
      return (*(dynamic_cast<LibPortBus*>(p->second)))[index.value()];
    }
  }

  return nullptr;
}

/**
 * @brief Find the liberty arc match from port name and to port name.
 *
 * @param from_port_name
 * @param to_port_name
 * @param timing_type
 * @return unsigned 1 if success, 0 else fail.
 */
std::optional<LibArcSet*> LibCell::findLibertyArcSet(const char* from_port_name, const char* to_port_name, LibArc::TimingType timing_type)
{
  for (auto& cell_arc_set : _cell_arcs) {
    auto* cell_arc = cell_arc_set->front();

    if (Str::equal(from_port_name, cell_arc->get_src_port()) && Str::equal(to_port_name, cell_arc->get_snk_port())
        && (timing_type == cell_arc->get_timing_type())) {
      return cell_arc_set.get();
    }
  }

  return std::nullopt;
}

/**
 * @brief Find the liberty arc match from port name and to port name.
 *
 * @param from_port_name
 * @param to_port_name
 * @return unsigned 1 if success, 0 else fail.
 */
std::optional<LibArcSet*> LibCell::findLibertyArcSet(const char* from_port_name, const char* to_port_name)
{
  for (auto& cell_arc_set : _cell_arcs) {
    auto* cell_arc = cell_arc_set->front();

    if (Str::equal(from_port_name, cell_arc->get_src_port()) && Str::equal(to_port_name, cell_arc->get_snk_port())) {
      return cell_arc_set.get();
    }
  }

  return std::nullopt;
}

/**
 * @brief Find the liberty arc set of to port name.
 *
 * @param to_port_name
 * @return std::vector<LibArcSet*>
 */
std::vector<LibArcSet*> LibCell::findLibertyArcSet(const char* to_port_name)
{
  std::vector<LibArcSet*> ret_value;
  for (auto& cell_arc_set : _cell_arcs) {
    auto* cell_arc = cell_arc_set->front();

    if (Str::equal(to_port_name, cell_arc->get_snk_port())) {
      ret_value.emplace_back(cell_arc_set.get());
    }
  }

  return ret_value;
}

/**
 * @brief Find the liberty arc match from port name and to port name.
 *
 * @param from_port_name
 * @param to_port_name
 * @return unsigned 1 if success, 0 else fail.
 */
std::optional<LibPowerArcSet*> LibCell::findLibertyPowerArcSet(const char* from_port_name, const char* to_port_name)
{
  for (auto& cell_power_arc_set : _cell_power_arcs) {
    auto* cell_power_arc = cell_power_arc_set->front();

    if (Str::equal(from_port_name, cell_power_arc->get_src_port()) && Str::equal(to_port_name, cell_power_arc->get_snk_port())) {
      return cell_power_arc_set.get();
    }
  }

  return std::nullopt;
}

/**
 * @brief Get the buffer port input port and output port.
 *
 * @param input
 * @param output
 */
void LibCell::bufferPorts(LibPort*& input, LibPort*& output)
{
  input = nullptr;
  output = nullptr;
  for (auto& port : _cell_ports) {
    if (port->isInput()) {
      if (input) {
        // More than one input.
        input = nullptr;
        output = nullptr;
        break;
      }
      input = port.get();
    } else if (port->isOutput()) {
      if (output) {
        // More than one output.
        input = nullptr;
        output = nullptr;
        break;
      }
      output = port.get();
    }
  }
}

/**
 * @brief Judge cell is buffer.
 *
 * @param input
 * @param output
 * @return true
 * @return false
 */
bool LibCell::hasBufferFunc(LibPort* input, LibPort* output)
{
  auto* func_expr = output->get_func_expr();
  return func_expr && func_expr->op == RustLibertyExprOp::kBuffer;
}

/**
 * @brief Judge cell is inverter.
 *
 * @param input
 * @param output
 * @return true
 * @return false
 */
bool LibCell::hasInverterFunc(LibPort* input, LibPort* output)
{
  auto* func_expr = output->get_func_expr();
  return func_expr && func_expr->op == RustLibertyExprOp::kNot;
}

/**
 * @brief Judge the buffer is cell.
 *
 * @return true
 * @return false
 */
bool LibCell::isBuffer()
{
  LibPort* input;
  LibPort* output;
  bufferPorts(input, output);
  return input && output && hasBufferFunc(input, output);
}

/**
 * @brief Judge the buffer is inverter.
 *
 * @return true
 * @return false
 */
bool LibCell::isInverter()
{
  LibPort* input;
  LibPort* output;
  bufferPorts(input, output);
  return input && output && hasInverterFunc(input, output);
}

/**
 * @brief judge whether cell is seq cell.
 * @return true
 * @return false
 */
bool LibCell::isSequentialCell()
{
  for (auto& liberty_arc_set : _cell_arcs) {
    auto& lib_arc = liberty_arc_set->get_arcs().front();
    if (lib_arc->isCheckArc()) {
      return true;
    }
  }
  return false;
}

/**
 * @brief judge whether the cell is ICG(Intergrated Clock Gating Cell).
 * @return true
 * @return false
 */
bool LibCell::isICG()
{
  for (auto& liberty_arc_set : _cell_arcs) {
    auto& lib_arc = liberty_arc_set->get_arcs().front();
    if (lib_arc->isClockGateCheckArc()) {
      return true;
    }
  }
  return false;
}

/**
 * @brief convert table power to mw unit.
 *
 * @param query_table_power
 * @return double
 */
double LibCell::convertTablePowerToMw(double query_table_power)
{
  auto* the_lib = get_owner_lib();

  double power_mw = query_table_power;
  if (the_lib->get_cap_unit() == CapacitiveUnit::kFF) {
    power_mw = query_table_power / 1000.0;  // convert to mW.
  }

  return power_mw;
}

LibWireLoad::LibWireLoad(const char* wire_load_name) : _wire_load_name(wire_load_name)
{
}

LibLutTableTemplate::LibLutTableTemplate(const char* template_name) : _template_name(template_name)
{
}

const std::map<std::string_view, LibLutTableTemplate::Variable> LibLutTableTemplate::_str2var
    = {{"total_output_net_capacitance", LibLutTableTemplate::Variable::TOTAL_OUTPUT_NET_CAPACITANCE},
       {"input_net_transition", LibLutTableTemplate::Variable::INPUT_NET_TRANSITION},
       {"related_pin_transition", LibLutTableTemplate::Variable::RELATED_PIN_TRANSITION},
       {"constrained_pin_transition", LibLutTableTemplate::Variable::CONSTRAINED_PIN_TRANSITION},
       {"input_transition_time", LibLutTableTemplate::Variable::INPUT_TRANSITION_TIME},
       {"time", LibLutTableTemplate::Variable::TIME},
       {"input_voltage", LibLutTableTemplate::Variable::INPUT_VOLTAGE},
       {"output_voltage", LibLutTableTemplate::Variable::OUTPUT_VOLTAGE},
       {"input_noise_height", LibLutTableTemplate::Variable::INPUT_NOISE_HEIGHT},
       {"input_noise_width", LibLutTableTemplate::Variable::INPUT_NOISE_WIDTH},
       {"normalized_voltage", LibLutTableTemplate::Variable::NORMALIZED_VOLTAGE},
       {"equal_or_opposite_output_net_capacitance", LibLutTableTemplate::Variable::EQUAL_OR_OPPOSITE_OUTPUT_NET_CAPACITANCE}};

LibCurrentTemplate::LibCurrentTemplate(const char* template_name) : LibLutTableTemplate(template_name)
{
}

/**
 * @brief print the LibertyLibrary in json format.
 *
 */
void LibLibrary::printLibertyLibraryJson(const char* json_file_name)
{
  auto create_timing_arc = [](LibArc* lib_arc) {
    nlohmann::json timing_arc = nlohmann::json::object();
    timing_arc["source_sink"] = {lib_arc->get_src_port(), lib_arc->get_snk_port()};
    LibTableModel* table_model = lib_arc->get_table_model();
    LibDelayTableModel* delay_model = dynamic_cast<LibDelayTableModel*>(table_model);

    // cell_rise table
    LibTable* cell_rise_table = dynamic_cast<LibDelayTableModel*>(table_model)->getTable(int(LibTable::TableType::kCellRise));
    if (cell_rise_table) {
      auto& axes = cell_rise_table->get_axes();
      int rows = axes[0].get()->get_axis_values().size();
      int columns = axes[1].get()->get_axis_values().size();
      nlohmann::json cell_rise_data;
      for (int i = 0; i < axes.size(); i++) {
        auto& axis_values = axes[i].get()->get_axis_values();
        nlohmann::json index = nlohmann::json::array();
        for (int j = 0; j < axis_values.size(); ++j) {
          auto axis_float_value = dynamic_cast<LibFloatValue*>(axis_values[j].get())->getFloatValue();
          index.push_back(axis_float_value);
        }
        // index_1
        // ("0.00117378,0.00472397,0.0171859,0.0409838,0.0780596,0.130081,0.198535");
        cell_rise_data["index_" + std::to_string(i + 1)] = index;
      }
      auto& lib_table_values = cell_rise_table->get_table_values();
      nlohmann::json values_array = nlohmann::json::array();
      for (size_t i = 0; i < lib_table_values.size(); i += columns) {
        nlohmann::json row = nlohmann::json::array();
        for (size_t j = 0; j < columns && (i + j) < lib_table_values.size(); ++j) {
          auto lib_table_float_value = dynamic_cast<LibFloatValue*>(lib_table_values[i + j].get())->getFloatValue();
          row.push_back(lib_table_float_value);
        }
        values_array.push_back(row);
      }
      cell_rise_data["values"] = values_array;
      timing_arc["cell_rise"] = cell_rise_data;
    }

    // rise_transition table
    LibTable* rise_transition_table = dynamic_cast<LibDelayTableModel*>(table_model)->getTable(int(LibTable::TableType::kRiseTransition));
    if (rise_transition_table) {
      auto& rise_trans_axes = rise_transition_table->get_axes();
      int rise_trans_rows = rise_trans_axes[0].get()->get_axis_values().size();
      int rise_trans_columns = rise_trans_axes[1].get()->get_axis_values().size();
      nlohmann::json rise_transition_data;
      for (int i = 0; i < rise_trans_axes.size(); i++) {
        auto& axis_values = rise_trans_axes[i].get()->get_axis_values();
        nlohmann::json index = nlohmann::json::array();
        for (int j = 0; j < axis_values.size(); ++j) {
          auto axis_float_value = dynamic_cast<LibFloatValue*>(axis_values[j].get())->getFloatValue();
          index.push_back(axis_float_value);
        }
        rise_transition_data["index_" + std::to_string(i + 1)] = index;
      }
      auto& rise_trans_lib_table_values = rise_transition_table->get_table_values();
      nlohmann::json rise_trans_values_array = nlohmann::json::array();
      for (size_t i = 0; i < rise_trans_lib_table_values.size(); i += rise_trans_columns) {
        nlohmann::json row = nlohmann::json::array();
        for (size_t j = 0; j < rise_trans_columns && (i + j) < rise_trans_lib_table_values.size(); ++j) {
          auto lib_table_float_value = dynamic_cast<LibFloatValue*>(rise_trans_lib_table_values[i + j].get())->getFloatValue();
          row.push_back(lib_table_float_value);
        }
        rise_trans_values_array.push_back(row);
      }
      rise_transition_data["values"] = rise_trans_values_array;
      timing_arc["rise_transition"] = rise_transition_data;
    }

    // cell_fall table
    LibTable* cell_fall_table = dynamic_cast<LibDelayTableModel*>(table_model)->getTable(int(LibTable::TableType::kCellFall));
    if (cell_fall_table) {
      auto& cell_fall_axes = cell_fall_table->get_axes();
      int cell_fall_rows = cell_fall_axes[0].get()->get_axis_values().size();
      int cell_fall_columns = cell_fall_axes[1].get()->get_axis_values().size();
      nlohmann::json cell_fall_data;
      for (int i = 0; i < cell_fall_axes.size(); i++) {
        auto& axis_values = cell_fall_axes[i].get()->get_axis_values();
        nlohmann::json index = nlohmann::json::array();
        for (int j = 0; j < axis_values.size(); ++j) {
          auto axis_float_value = dynamic_cast<LibFloatValue*>(axis_values[j].get())->getFloatValue();
          index.push_back(axis_float_value);
        }
        cell_fall_data["index_" + std::to_string(i + 1)] = index;
      }
      auto& cell_fall_lib_table_values = cell_fall_table->get_table_values();
      nlohmann::json cell_fall_values_array = nlohmann::json::array();
      for (size_t i = 0; i < cell_fall_lib_table_values.size(); i += cell_fall_columns) {
        nlohmann::json row = nlohmann::json::array();
        for (size_t j = 0; j < cell_fall_columns && (i + j) < cell_fall_lib_table_values.size(); ++j) {
          auto lib_table_float_value = dynamic_cast<LibFloatValue*>(cell_fall_lib_table_values[i + j].get())->getFloatValue();
          row.push_back(lib_table_float_value);
        }
        cell_fall_values_array.push_back(row);
      }
      cell_fall_data["values"] = cell_fall_values_array;
      timing_arc["cell_fall"] = cell_fall_data;
    }

    // fall_transition table
    LibTable* fall_transition_table = dynamic_cast<LibDelayTableModel*>(table_model)->getTable(int(LibTable::TableType::kFallTransition));
    if (fall_transition_table) {
      auto& fall_transition_axes = fall_transition_table->get_axes();
      int fall_transition_rows = fall_transition_axes[0].get()->get_axis_values().size();
      int fall_transition_columns = fall_transition_axes[1].get()->get_axis_values().size();
      nlohmann::json fall_transition_data;
      for (int i = 0; i < fall_transition_axes.size(); i++) {
        auto& axis_values = fall_transition_axes[i].get()->get_axis_values();
        nlohmann::json index = nlohmann::json::array();
        for (int j = 0; j < axis_values.size(); ++j) {
          auto axis_float_value = dynamic_cast<LibFloatValue*>(axis_values[j].get())->getFloatValue();
          index.push_back(axis_float_value);
        }
        fall_transition_data["index_" + std::to_string(i + 1)] = index;
      }
      auto& fall_transition_lib_table_values = fall_transition_table->get_table_values();
      nlohmann::json fall_transition_values_array = nlohmann::json::array();
      for (size_t i = 0; i < fall_transition_lib_table_values.size(); i += fall_transition_columns) {
        nlohmann::json row = nlohmann::json::array();
        for (size_t j = 0; j < fall_transition_columns && (i + j) < fall_transition_lib_table_values.size(); ++j) {
          auto lib_table_float_value = dynamic_cast<LibFloatValue*>(fall_transition_lib_table_values[i + j].get())->getFloatValue();
          row.push_back(lib_table_float_value);
        }
        fall_transition_values_array.push_back(row);
      }
      fall_transition_data["values"] = fall_transition_values_array;
      timing_arc["fall_transition"] = fall_transition_data;
    }

    return timing_arc;
  };

  auto classify_cell_arc_by_snk_port = [](LibCell* lib_cell) -> std::map<std::string, std::vector<LibArc*>> {
    std::map<std::string, std::vector<LibArc*>> snkport2arcset;
    for (auto& cell_arc_set : lib_cell->get_cell_arcs()) {
      auto* cell_arc = cell_arc_set->front();
      const char* src_port_name = cell_arc->get_src_port();
      const char* snk_port_name = cell_arc->get_snk_port();
      snkport2arcset[snk_port_name].push_back(cell_arc);
    }

    return snkport2arcset;
  };

  nlohmann::json json_data;
  json_data["lib_name"] = get_lib_name();
  for (const auto& cell : get_cells()) {
    nlohmann::json cell_info;
    cell_info["cell_name"] = cell->get_cell_name();
    cell_info["timing_arcs"] = nlohmann::json::array();

    auto snkport2arcset = classify_cell_arc_by_snk_port(cell.get());
    for (const auto& pair : snkport2arcset) {
      for (const auto& arc : pair.second) {
        if (arc->isDelayArc()) {
          // wirte json
          cell_info["timing_arcs"].push_back(create_timing_arc(arc));
        }
      }
    }
    json_data["cells_lib_info"].push_back(cell_info);
  }

  std::ofstream json_file(json_file_name);
  if (json_file.is_open()) {
    LOG_INFO << "start write liberty into json file: " << json_file_name;
    json_file << json_data.dump(1);
    json_file.close();
    LOG_INFO << "success write liberty into json file: " << json_file_name;
  } else {
    LOG_INFO << "fail write liberty into json file: " << json_file_name;
  }
}

/**
 * @brief Load liberty with rust parse API.
 *
 * @param file_name
 * @return std::unique_ptr<LibLibrary>
 */
RustLibertyReader Lib::loadLibertyWithRustParser(const char* file_name)
{
  // LOG_INFO << "Load lib " << file_name << " start.";

  RustLibertyReader lib_rust_reader(file_name);
  unsigned is_success = lib_rust_reader.readLib();
  LOG_FATAL_IF(!is_success) << "read lib " << file_name << " failed.";

  // LOG_INFO << "Load lib " << file_name << " finish.";
  return lib_rust_reader;
}

}  // namespace ista
