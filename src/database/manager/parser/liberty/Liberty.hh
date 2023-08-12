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
 * @file Liberty.h
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief This is the interface of liberty module.
 * @version 0.1
 * @date 2020-11-28
 */

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "Array.hh"
#include "HashMap.hh"
#include "HashSet.hh"
#include "Map.hh"
#include "Vector.hh"
#include "include/Config.hh"
#include "include/Type.hh"
#include "log/Log.hh"
#include "mLibertyExpr.hh"
#include "string/Str.hh"
#include "string/StrMap.hh"

namespace ista {

class LibertyType;
class LibertyCell;
class LibertyLibrary;
class LibertyAttrValue;
class LibertyAxis;
class LibertyLutTableTemplate;
class LibertyVectorTable;
class LibertyExpr;
class LibertyCellPortIterator;
class LibertyCellIterator;
class LibertyCellTimingArcSetIterator;
class LibertyCellPowerArcSetIterator;

/**
 * @brief The base object of the library.
 *
 */
class LibertyObject
{
 public:
  LibertyObject() = default;
  virtual ~LibertyObject() = default;

  virtual void addAxis(std::unique_ptr<LibertyAxis>&& axis) { LOG_FATAL << "not support"; }
  virtual void set_template_variable1(const char*) { LOG_FATAL << "not support"; }
  virtual void set_template_variable2(const char*) { LOG_FATAL << "not support"; }
  virtual void set_template_variable3(const char*) { LOG_FATAL << "not support"; }

  virtual void set_template_variable4(const char*) { LOG_FATAL << "not support"; }

  virtual unsigned isLibertyPortBus() { return 0; }

  void set_file_name(const char* file_name) { _file_name = file_name; }
  const char* get_file_name() { return _file_name.c_str(); }

  void set_line_no(unsigned line_no) { _line_no = line_no; }
  [[nodiscard]] unsigned get_line_no() const { return _line_no; }

 private:
  std::string _file_name;
  unsigned _line_no = 0;

  DISALLOW_COPY_AND_ASSIGN(LibertyObject);
};

/**
 * @brief The liberty axis, which the lut table consist of.
 *
 */
class LibertyAxis : public LibertyObject
{
 public:
  explicit LibertyAxis(const char* axis_name);
  ~LibertyAxis() override = default;

  LibertyAxis(LibertyAxis&& other) noexcept;
  LibertyAxis& operator=(LibertyAxis&& rhs) noexcept;

  const char* get_axis_name() { return _axis_name.c_str(); }

  void set_axis_values(std::vector<std::unique_ptr<LibertyAttrValue>>&& table_values) { _axis_values = std::move(table_values); }

  auto& get_axis_values() { return _axis_values; }
  std::size_t get_axis_size() { return _axis_values.size(); }

  double operator[](std::size_t index);

 private:
  std::string _axis_name;  //!< The axis name.

  std::vector<std::unique_ptr<LibertyAttrValue>> _axis_values;  //!< The axis sample values.

  DISALLOW_COPY_AND_ASSIGN(LibertyAxis);
};

/**
 * @brief The liberty NLDM table.
 *
 */
class LibertyTable : public LibertyObject
{
 public:
  enum class TableType : int
  {
    kCellRise = 0,
    kCellFall = 1,
    kRiseTransition = 2,
    kFallTransition = 3,
    kRiseConstrain = 4,
    kFallConstrain = 5,
    kRiseCurrent = 6,
    kFallCurrent = 7,
    // power
    kRisePower = 8,
    kFallPower = 9
  };

  static const std::map<std::string, TableType> _str2TableType;
  static const unsigned _time_index = 2;

  LibertyTable(TableType table_type, LibertyLutTableTemplate* table_template);

  ~LibertyTable() override = default;

  LibertyTable(LibertyTable&& other) noexcept;
  LibertyTable& operator=(LibertyTable&& rhs) noexcept;

  void addAxis(std::unique_ptr<LibertyAxis>&& axis) override { _axes.push_back(std::move(axis)); }

  LibertyAxis& getAxis(unsigned int index);

  Vector<std::unique_ptr<LibertyAxis>>& get_axes();

  void set_table_values(std::vector<std::unique_ptr<LibertyAttrValue>>&& table_values) { _table_values = std::move(table_values); }
  auto& get_table_values() { return _table_values; }

  TableType get_table_type() { return _table_type; }

  void set_table_template(LibertyLutTableTemplate* table_template) { _table_template = table_template; }
  LibertyLutTableTemplate* get_table_template() { return _table_template; }

  double findValue(double slew, double constrain_slew_or_load);

  double driveResistance();

 private:
  Vector<std::unique_ptr<LibertyAxis>> _axes;                    //!< May be zero, one, two, three axes.
  std::vector<std::unique_ptr<LibertyAttrValue>> _table_values;  //!< The axis values.
  TableType _table_type;                                         //!< The table type.

  LibertyLutTableTemplate* _table_template;  //!< The lut template.

  DISALLOW_COPY_AND_ASSIGN(LibertyTable);
};

/**
 * @brief The CCS model simulation information.
 *
 */
struct LibertyCurrentSimuInfo
{
  double _start_time;
  double _end_time;
  int _num_sim_point;
};

/**
 * @brief The current vector table, such as:
 *
 * vector(Output_current) {
 *         reference_time : 3.115910;
 *         index_1 ("0.00472397");
 *         index_2 ("0.365616");
 *         index_3("3.154048,3.154681,3.157112,3.159775,3.163000,3.165700,3.167170,3.168740,3.171171");
 *         values("-0.00616907,-0.00724877,-0.0171559,-0.0279607,-0.0365291,-0.0398586,-0.0304221,-0.00932126,-0.00135837");
 * }
 */
class LibertyVectorTable : public LibertyTable
{
 public:
  LibertyVectorTable(TableType table_type, LibertyLutTableTemplate* table_template);
  ~LibertyVectorTable() override = default;

  LibertyVectorTable(LibertyVectorTable&& other) noexcept;
  LibertyVectorTable& operator=(LibertyVectorTable&& rhs) noexcept;

  void set_ref_time(double ref_time) { _ref_time = ref_time; }
  [[nodiscard]] double get_ref_time() const { return _ref_time; }

  std::tuple<double, int> getSimulationTotalTimeAndNumPoints();

  std::vector<double> getOutputCurrent(std::optional<LibertyCurrentSimuInfo>& simu_info);

 private:
  double _ref_time = 0.0;  //!< The current reference time.

  DISALLOW_COPY_AND_ASSIGN(LibertyVectorTable);
};

/**
 * @brief The output current data for upper layer interface.
 *
 */
class LibetyCurrentData
{
 public:
  LibetyCurrentData(LibertyVectorTable* low_low, LibertyVectorTable* low_high, LibertyVectorTable* high_low, LibertyVectorTable* high_high,
                    double slew, double load);
  ~LibetyCurrentData() = default;

  LibetyCurrentData(const LibetyCurrentData& orig) = default;
  LibetyCurrentData& operator=(const LibetyCurrentData& rhs) = default;

  LibetyCurrentData(LibetyCurrentData&& orig) = default;
  LibetyCurrentData& operator=(LibetyCurrentData&& rhs) = default;

  LibetyCurrentData* copy() { return new LibetyCurrentData(*this); }

  LibertyVectorTable* get_low_low() { return _low_low; }
  LibertyVectorTable* get_low_high() { return _low_high; }
  LibertyVectorTable* get_high_low() { return _high_low; }
  LibertyVectorTable* get_high_high() { return _high_high; }

  std::tuple<double, int> getSimulationTotalTimeAndNumPoints();

  std::vector<double> getOutputCurrent(std::optional<LibertyCurrentSimuInfo>& simu_info);

 private:
  LibertyVectorTable* _low_low;    //!< low slew and low load
  LibertyVectorTable* _low_high;   //!< low slew and high load
  LibertyVectorTable* _high_low;   //!< high slew and low load
  LibertyVectorTable* _high_high;  //!< high slew and high load

  double _slew;
  double _load;
};

/**
 * @brief The liberty CCS table include one or more vector table.
 *
 */
class LibertyCCSTable : public LibertyObject
{
 public:
  explicit LibertyCCSTable(LibertyTable::TableType table_type);
  ~LibertyCCSTable() override = default;
  void addTable(std::unique_ptr<LibertyVectorTable>&& current_table) { _vector_tables.emplace_back(std::move(current_table)); }
  auto get_table_type() { return _table_type; }
  auto& get_vector_tables() { return _vector_tables; }

 private:
  LibertyTable::TableType _table_type;                              //!< The table type.
  std::vector<std::unique_ptr<LibertyVectorTable>> _vector_tables;  //!< The current tables.

  DISALLOW_COPY_AND_ASSIGN(LibertyCCSTable);
};

#define STR_TO_TABLE_TYPE(str) LibertyTable::_str2TableType.at(str)

/**
 * @brief The liberty table model, include delay model and check model.
 *
 */
class LibertyTableModel : public LibertyObject
{
 public:
  LibertyTableModel() = default;
  ~LibertyTableModel() override = default;
  virtual unsigned isDelayModel() { return 0; }
  virtual unsigned isCheckModel() { return 0; }
  virtual unsigned isPowerModel() { return 0; }
  virtual unsigned addTable(std::unique_ptr<LibertyTable>&& table) = 0;
  virtual LibertyTable* getTable(int index) = 0;
  virtual double gateDelay(TransType trans_type, double slew, double load)
  {
    LOG_FATAL << "not support";
    return 0.0;
  }
  virtual double gateSlew(TransType trans_type, double slew, double load)
  {
    LOG_FATAL << "not support";
    return 0.0;
  }
  virtual double gateCheckConstrain(TransType trans_type, double slew, double load)
  {
    LOG_FATAL << "not support";
    return 0.0;
  }

  virtual std::unique_ptr<LibetyCurrentData> gateOutputCurrent(TransType trans_type, double slew, double load)
  {
    LOG_FATAL << "not support";
    return nullptr;
  }

  virtual double driveResistance() { return 0.0; }

  virtual double gatePower(TransType trans_type, double slew, std::optional<double> load)
  {
    LOG_FATAL << "not support";
    return 0.0;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(LibertyTableModel);
};

#define CAST_TYPE_TO_INDEX(type) ((static_cast<int>(type) > 3) ? (static_cast<int>(type) - 4) : static_cast<int>(type))
#define CAST_CURRENT_TYPE_TO_INDEX(type) (static_cast<int>(type) - 6)
#define CAST_POWER_TYPE_TO_INDEX(type) (static_cast<int>(type) - 8)

/**
 * @brief The liberty delay model.
 *
 */
class LibertyDelayTableModel final : public LibertyTableModel
{
 public:
  static constexpr size_t kTableNum = 4;         //!< The model contain delay/slew, rise/fall four table.
  static constexpr size_t kCurrentTableNum = 2;  //!< Current rise/fall table.

  unsigned isDelayModel() override { return 1; }

  LibertyDelayTableModel() = default;
  ~LibertyDelayTableModel() = default;

  LibertyDelayTableModel(LibertyDelayTableModel&& other) noexcept;
  LibertyDelayTableModel& operator=(LibertyDelayTableModel&& rhs) noexcept;

  unsigned addTable(std::unique_ptr<LibertyTable>&& table)
  {
    auto table_type = table->get_table_type();
    _tables[CAST_TYPE_TO_INDEX(table_type)] = std::move(table);
    return 1;
  }

  LibertyTable* getTable(int index) override { return _tables[index].get(); }

  unsigned addCurrentTable(std::unique_ptr<LibertyCCSTable>&& table)
  {
    auto table_type = table->get_table_type();
    _current_tables[CAST_CURRENT_TYPE_TO_INDEX(table_type)] = std::move(table);
    return 1;
  }

  double gateDelay(TransType trans_type, double slew, double load) override;
  double gateSlew(TransType trans_type, double slew, double load) override;
  std::unique_ptr<LibetyCurrentData> gateOutputCurrent(TransType trans_type, double slew, double load) override;

  double driveResistance() override;

 private:
  std::array<std::unique_ptr<LibertyTable>, kTableNum> _tables;  // NLDM table,include cell rise/cell fall/rise transition/fall
                                                                 // transition.
  std::array<std::unique_ptr<LibertyCCSTable>,
             kCurrentTableNum>  // Output current rise/fall.
      _current_tables;

  DISALLOW_COPY_AND_ASSIGN(LibertyDelayTableModel);
};

/**
 * @brief The liberty check model.
 *
 */
class LibertyCheckTableModel final : public LibertyTableModel
{
 public:
  static constexpr size_t kTableNum = 2;  //!< The model contain rise/fall constrain two tables.

  unsigned isCheckModel() override { return 1; }

  LibertyCheckTableModel() = default;
  ~LibertyCheckTableModel() override = default;

  LibertyCheckTableModel(LibertyCheckTableModel&& other) noexcept;
  LibertyCheckTableModel& operator=(LibertyCheckTableModel&& rhs) noexcept;

  unsigned addTable(std::unique_ptr<LibertyTable>&& table) override
  {
    auto table_type = table->get_table_type();
    _tables[CAST_TYPE_TO_INDEX(table_type)] = std::move(table);
    return 1;
  }

  LibertyTable* getTable(int index) override { return _tables[index].get(); }
  double gateCheckConstrain(TransType trans_type, double slew, double constrain_slew) override;

 private:
  std::array<std::unique_ptr<LibertyTable>, kTableNum> _tables;

  DISALLOW_COPY_AND_ASSIGN(LibertyCheckTableModel);
};

/**
 * @brief The liberty power model.
 *
 */
class LibertyPowerTableModel final : public LibertyTableModel
{
 public:
  static constexpr size_t kTableNum = 2;  //!< The model contain rise/fall power two table.
  unsigned isPowerModel() override { return 1; }

  LibertyPowerTableModel() = default;
  ~LibertyPowerTableModel() override = default;

  LibertyPowerTableModel(LibertyPowerTableModel&& other) noexcept;
  LibertyPowerTableModel& operator=(LibertyPowerTableModel&& rhs) noexcept;

  unsigned addTable(std::unique_ptr<LibertyTable>&& table)
  {
    auto table_type = table->get_table_type();
    _tables[CAST_POWER_TYPE_TO_INDEX(table_type)] = std::move(table);
    return 1;
  }
  LibertyTable* getTable(int index) override { return _tables[index].get(); }

  double gatePower(TransType trans_type, double slew, std::optional<double> load) override;

 private:
  std::array<std::unique_ptr<LibertyTable>, kTableNum> _tables;  // power table,include rise power/fall power.
  DISALLOW_COPY_AND_ASSIGN(LibertyPowerTableModel);
};

/**
 * @brief class for internal power information
 *
 */
class LibertyInternalPowerInfo
{
 public:
  void set_related_pg_port(const char* related_pg_port) { _related_pg_port = related_pg_port; }
  auto& get_related_pg_port() { return _related_pg_port; }

  void set_when(const char* when) { _when = when; }
  auto& get_when() { return _when; }

  void set_power_table_model(std::unique_ptr<LibertyTableModel>&& power_table_model) { _power_table_model = std::move(power_table_model); }
  LibertyTableModel* get_power_table_model() { return _power_table_model.get(); }

  double gatePower(TransType trans_type, double slew, std::optional<double> load)
  {
    return _power_table_model->gatePower(trans_type, slew, load);
  }

 private:
  std::string _related_pg_port;                           //!< The liberty power arc related pg port.
  std::string _when;                                      //!< The liberty power arc related pg port.
  std::unique_ptr<LibertyTableModel> _power_table_model;  //!< The pin power table model.
};

/**
 * @brief The port in the cell.
 *
 */
class LibertyPort : public LibertyObject
{
 public:
  enum class LibertyPortType
  {
    kDefault = 0,
    kInput = 1,
    kOutput = 2,
    kInOut = 3
  };
  enum class LibertyCapIndex : int
  {
    kMaxRise = 0,
    kMaxFall = 1,
    kMinRise = 2,
    kMinFall = 3
  };

  enum class LibertyMaxMinLimitIndex : int
  {
    kMax = 0,
    kMin = 1
  };

  explicit LibertyPort(const char* port_name);
  ~LibertyPort() override = default;

  LibertyPort(LibertyPort&& other) noexcept;
  LibertyPort& operator=(LibertyPort&& rhs) noexcept;

  const char* get_port_name() { return _port_name.c_str(); }
  void set_ower_cell(LibertyCell* ower_cell) { _ower_cell = ower_cell; }
  LibertyCell* get_ower_cell() { return _ower_cell; }

  void set_port_type(LibertyPortType port_type) { _port_type = port_type; }
  LibertyPortType get_port_type() { return _port_type; }

  void set_clock_gate_clock_pin(bool clock_gate_clock_pin) { _clock_gate_clock_pin = clock_gate_clock_pin; }
  bool get_clock_gate_clock_pin() { return _clock_gate_clock_pin; }

  void set_clock_gate_enable_pin(bool clock_gate_enable_pin) { _clock_gate_enable_pin = clock_gate_enable_pin; }
  bool get_clock_gate_enable_pin() { return _clock_gate_enable_pin; }

  void set_port_cap(double cap) { _port_cap = cap; }
  double get_port_cap() const { return _port_cap; }

  void set_port_cap(AnalysisMode mode, TransType trans_type, double cap);
  std::optional<double> get_port_cap(AnalysisMode mode, TransType trans_type);

  void set_port_cap_limit(AnalysisMode mode, double cap_limit);
  std::optional<double> get_port_cap_limit(AnalysisMode mode);

  void set_port_slew_limit(AnalysisMode mode, double slew_limit);
  std::optional<double> get_port_slew_limit(AnalysisMode mode);

  void set_func_expr(LibertyExpr* lib_expr);
  LibertyExpr* get_func_expr();

  void set_func_expr_str(const char* func_expr_str) { _func_expr_str = func_expr_str; }
  auto& get_func_expr_str() { return _func_expr_str; }

  unsigned isInput() { return (_port_type == LibertyPortType::kInput || _port_type == LibertyPortType::kInOut); }
  unsigned isOutput() { return (_port_type == LibertyPortType::kOutput || _port_type == LibertyPortType::kInOut); }
  unsigned isInout() { return _port_type == LibertyPortType::kInOut; }

  void set_port_type(const char* port_type)
  {
    if (Str::equal(port_type, "input")) {
      _port_type = LibertyPortType::kInput;
    } else if (Str::equal(port_type, "output")) {
      _port_type = LibertyPortType::kOutput;
    } else {
      _port_type = LibertyPortType::kInOut;
    }
  }

  void set_fanout_load(double fanout_load_val) { _fanout_load = fanout_load_val; }
  auto& get_fanout_load() { return _fanout_load; }

  double driveResistance();

  bool isClock();
  bool isSeqDataIn();

  void addInternalPower(std::unique_ptr<LibertyInternalPowerInfo>&& internal_power)
  {
    _internal_powers.emplace_back(std::move(internal_power));
  }
  auto& get_internal_powers() { return _internal_powers; }

 private:
  std::string _port_name;
  LibertyCell* _ower_cell;  //!< The cell owner the port.
  LibertyPortType _port_type = LibertyPortType::kDefault;
  bool _clock_gate_clock_pin = false;   //!< The flag of gate clock pin.
  bool _clock_gate_enable_pin = false;  //!< The flag of gate enable pin.
  std::unique_ptr<LibertyExpr> _func_expr;
  std::string _func_expr_str;                                        //!< store func expr string for debug.
  double _port_cap = 0.0;                                            //!< The input pin corresponding to the port has capacitance.
  std::array<std::optional<double>, MODE_TRANS_SPLIT> _port_caps{};  //!< May be port cap split max rise, max fall, min rise,
                                                                     //!< min fall.
  std::array<std::optional<double>, MODE_SPLIT> _cap_limits{};
  std::array<std::optional<double>, MODE_SPLIT> _slew_limits{};

  std::optional<double> _fanout_load;

  Vector<std::unique_ptr<LibertyInternalPowerInfo>> _internal_powers;  //!< The internal power information.

  DISALLOW_COPY_AND_ASSIGN(LibertyPort);
};

/**
 * @brief The macro of foreach internal power, usage:
 * LibertyPort* port;
 * LibertyInternalPowerInfo* internal_power;
 * FOREACH_INTERNAL_POWER(port, internal_power)
 * {
 *    do_something_for_internal_power();
 * }
 */
#define FOREACH_INTERNAL_POWER(port, internal_power)                                   \
  if (auto& internal_powers = (port)->get_internal_powers(); !internal_powers.empty()) \
    for (auto p = internal_powers.begin(); p != internal_powers.end() ? internal_power = p->get(), true : false; ++p)

/**
 * @brief The liberty bus type class，such as:
 *   type (S011HD1P_X64Y4D32_BW_DATA) {
 *           base_type : array ;
 *           data_type : bit ;
 *           bit_width : 32;
 *           bit_from : 31;
 *           bit_to : 0 ;
 *           downto : true ;
 *       }
 */
class LibertyType : public LibertyObject
{
 public:
  explicit LibertyType(std::string&& type_name) : _type_name(std::move(type_name)) {}

  const char* get_type_name() { return _type_name.c_str(); }

  void set_base_type(std::string&& base_type) { _base_type = std::move(base_type); }
  auto& get_base_type() { return _base_type; }

  void set_data_type(std::string&& data_type) { _data_type = std::move(data_type); }
  auto& get_data_type() { return _data_type; }

  void set_bit_width(unsigned bit_width) { _bit_width = bit_width; }
  [[nodiscard]] unsigned get_bit_width() const { return _bit_width; }

  void set_bit_from(unsigned bit_from) { _bit_from = bit_from; }
  [[nodiscard]] unsigned get_bit_from() const { return _bit_from; }

  void set_bit_to(unsigned bit_to) { _bit_to = bit_to; }
  [[nodiscard]] unsigned get_bit_to() const { return _bit_to; }

 private:
  std::string _type_name;
  std::string _base_type;
  std::string _data_type;
  unsigned _bit_width = 0;
  unsigned _bit_from = 0;
  unsigned _bit_to = 0;
  bool _downto = false;
};

/**
 * @brief The port bus in the cell.
 *
 */
class LibertyPortBus : public LibertyPort
{
 public:
  explicit LibertyPortBus(const char* port_bus_name);
  ~LibertyPortBus() override = default;

  unsigned isLibertyPortBus() override { return 1; }

  void addlibertyPort(std::unique_ptr<LibertyPort>&& port) { _ports.push_back(std::move(port)); }

  auto getBusSize() { return _bus_type ? _bus_type->get_bit_width() : _ports.size(); }

  void set_bus_type(LibertyType* bus_type) { _bus_type = bus_type; }
  auto* get_bus_type() { return _bus_type; }

  LibertyPort* operator[](int index) { return _ports.empty() ? this : _ports[index].get(); }

 private:
  Vector<std::unique_ptr<LibertyPort>> _ports;  //!< The bus ports.
  LibertyType* _bus_type = nullptr;

  DISALLOW_COPY_AND_ASSIGN(LibertyPortBus);
};

/**
 * @brief The leakage power in the cell.
 *
 */
class LibertyLeakagePower : public LibertyObject
{
 public:
  LibertyLeakagePower();
  ~LibertyLeakagePower() override = default;

  LibertyLeakagePower(LibertyLeakagePower&& other) noexcept;
  LibertyLeakagePower& operator=(LibertyLeakagePower&& rhs) noexcept;

  void set_owner_cell(LibertyCell* ower_cell) { _owner_cell = ower_cell; }
  LibertyCell* get_owner_cell() { return _owner_cell; }
  void set_related_pg_port(const char* related_pg_port) { _related_pg_port = related_pg_port; }
  void set_when(const char* when) { _when = when; }
  void set_value(double value) { _value = value; }

  const char* get_related_pg_port() { return _related_pg_port.c_str(); }
  auto& get_when() { return _when; }
  double get_value() { return _value; }

 private:
  std::string _related_pg_port;  //!< The related pg pin of the leakage power.
  std::string _when;             //!< The when of the leakage power.
  double _value;                 //!< The value of the leakage power.
  LibertyCell* _owner_cell;      //!< The cell owner the port.

  DISALLOW_COPY_AND_ASSIGN(LibertyLeakagePower);
};

/**
 * @brief The timing arc in the liberty.
 *
 */
class LibertyArc : public LibertyObject
{
 public:
  enum class ArcType
  {
    kDelayArc,
    kCheckArc
  };
  enum class TimingSense
  {
    kPositiveUnate,
    kNegativeUnate,
    kNonUnate,
    kDefault
  };
  enum class TimingType : int
  {
    kSetupRising = 1,
    kHoldRising,
    kRecoveryRising,
    kRemovalRising,
    kRisingEdge,
    kPreset,
    kClear,
    kThreeStateEnable,
    kThreeStateEnableRise,
    kThreeStateEnableFall,
    kThreeStateDisable,
    kThreeStateDisableRise,
    kThreeStateDisableFall,
    kSetupFalling,
    kHoldFalling,
    kRecoveryFalling,
    kRemovalFalling,
    kFallingEdge,
    kMinPulseWidth,
    kCombRise,
    kCombFall,
    kComb,
    kNonSeqSetupRising,
    kNonSeqSetupFalling,
    kNonSeqHoldRising,
    kNonSeqHoldFalling,
    kSkewRising,
    kSkewFalling,
    kMinimunPeriod,
    kMaxClockTree,
    kMinClockTree,
    kNoChangeHighHigh,
    kNoChangeHighLow,
    kNoChangeLowHigh,
    kNoChangeLowLow,
    kDefault
  };

  LibertyArc();
  ~LibertyArc() override = default;

  LibertyArc(LibertyArc&& other) noexcept;
  LibertyArc& operator=(LibertyArc&& rhs) noexcept;

  void set_src_port(const char* src_port) { _src_port = src_port; }
  void set_snk_port(const char* snk_port) { _snk_port = snk_port; }
  const char* get_src_port() { return _src_port.c_str(); }
  const char* get_snk_port() { return _snk_port.c_str(); }

  void set_timing_sense(const char* timing_sense);
  TimingSense get_timing_sense() { return _timing_sense; }

  void set_timing_type(const char* timing_type);
  TimingType get_timing_type() { return _timing_type; }
  bool isMatchTimingType(TransType trans_type);

  void set_owner_cell(LibertyCell* ower_cell) { _owner_cell = ower_cell; }
  LibertyCell* get_owner_cell() { return _owner_cell; }

  unsigned isCheckArc();
  unsigned isDelayArc();
  unsigned isMpwArc();
  unsigned isClockGateCheckArc();
  unsigned isClearPresetArc() { return _timing_type == TimingType::kClear || _timing_type == TimingType::kPreset; }

  unsigned isUnateArc()
  {
    return (_timing_sense == TimingSense::kPositiveUnate) || (_timing_sense == TimingSense::kNegativeUnate)
           || (_timing_sense == TimingSense::kDefault);
  }

  unsigned isPositiveArc() { return _timing_sense == TimingSense::kPositiveUnate || (_timing_sense == TimingSense::kDefault); }

  unsigned isNegativeArc() { return _timing_sense == TimingSense::kNegativeUnate; }

  unsigned isSetupArc() { return (_timing_type == TimingType::kSetupRising) || (_timing_type == TimingType::kSetupFalling); }

  unsigned isHoldArc() { return (_timing_type == TimingType::kHoldRising) || (_timing_type == TimingType::kHoldFalling); }

  unsigned isRecoveryArc() { return (_timing_type == TimingType::kRecoveryRising) || (_timing_type == TimingType::kRecoveryFalling); }

  unsigned isRemovalArc() { return (_timing_type == TimingType::kRemovalRising) || (_timing_type == TimingType::kRemovalFalling); }

  unsigned isRisingEdgeCheck()
  {
    return (_timing_type == TimingType::kSetupRising) || (_timing_type == TimingType::kHoldRising)
           || (_timing_type == TimingType::kRecoveryRising) || (_timing_type == TimingType::kRemovalRising);
  }

  unsigned isFallingEdgeCheck()
  {
    return (_timing_type == TimingType::kSetupFalling) || (_timing_type == TimingType::kHoldFalling)
           || (_timing_type == TimingType::kRecoveryFalling) || (_timing_type == TimingType::kRemovalFalling);
  }

  unsigned isRisingTriggerArc() { return (_timing_type == TimingType::kRisingEdge); }

  unsigned isFallingTriggerArc() { return (_timing_type == TimingType::kFallingEdge); }

  void set_table_model(std::unique_ptr<LibertyTableModel>&& table_model) { _table_model = std::move(table_model); }
  LibertyTableModel* get_table_model() { return _table_model.get(); }

  double getDelayOrConstrainCheck(TransType trans_type, double slew, double load_or_constrain_slew);

  double getSlew(TransType trans_type, double slew, double load);

  std::unique_ptr<LibetyCurrentData> getOutputCurrent(TransType trans_type, double slew, double load);

  double getDriveResistance() { return _table_model->driveResistance(); }

 private:
  std::string _src_port;                           //!< The liberty timing arc source port, for liberty
                                                   //!< file port may be behind the arc, so we use port
                                                   //!< name, fix me.
  std::string _snk_port;                           //!< The liberty timing arc sink port.
  LibertyCell* _owner_cell;                        //!< The cell owner the port.
  TimingSense _timing_sense;                       //!< The arc timing sense.
  TimingType _timing_type = TimingType::kDefault;  //!< The arc timing type.

  std::unique_ptr<LibertyTableModel> _table_model;  //!< The arc timing model.

  static Map<std::string, TimingType> _str_to_type;

  DISALLOW_COPY_AND_ASSIGN(LibertyArc);
};

/**
 * @brief The liberty arc may have the same source and sink port, except the
 * condition is different.
 *
 */
class LibertyArcSet
{
 public:
  LibertyArcSet() = default;
  ~LibertyArcSet() = default;
  LibertyArcSet(LibertyArcSet&& other) noexcept;
  LibertyArcSet& operator=(LibertyArcSet&& rhs) noexcept;

  void addLibertyArc(std::unique_ptr<LibertyArc>&& cell_arc) { _arcs.emplace_back(std::move(cell_arc)); }

  LibertyArc* front() { return _arcs.front().get(); }
  auto& get_arcs() { return _arcs; }

 private:
  Vector<std::unique_ptr<LibertyArc>> _arcs;

  DISALLOW_COPY_AND_ASSIGN(LibertyArcSet);
};

/**
 * @brief The power arc in the liberty.
 *
 */
class LibertyPowerArc : public LibertyObject
{
 public:
  LibertyPowerArc();
  ~LibertyPowerArc() override = default;

  LibertyPowerArc(LibertyPowerArc&& other) noexcept;
  LibertyPowerArc& operator=(LibertyPowerArc&& rhs) noexcept;

  void set_src_port(const char* src_port) { _src_port = src_port; }
  void set_snk_port(const char* snk_port) { _snk_port = snk_port; }

  const char* get_src_port() { return _src_port.c_str(); }
  const char* get_snk_port() { return _snk_port.c_str(); }

  bool isSrcPortEmpty() { return _src_port.empty(); }
  bool isSnkPortEmpty() { return _snk_port.empty(); }

  void set_owner_cell(LibertyCell* ower_cell) { _owner_cell = ower_cell; }
  LibertyCell* get_owner_cell() { return _owner_cell; }

  void set_related_pg_port(const char* related_pg_port) { _internal_power_info->set_related_pg_port(related_pg_port); }
  const char* get_related_pg_port() { return _internal_power_info->get_related_pg_port().c_str(); }

  void set_when(const char* when) { _internal_power_info->set_when(when); }
  std::string get_when() { return _internal_power_info->get_when(); }

  void set_power_table_model(std::unique_ptr<LibertyTableModel>&& power_table_model)
  {
    _internal_power_info->set_power_table_model(std::move(power_table_model));
  }
  LibertyTableModel* get_power_table_model() { return _internal_power_info->get_power_table_model(); }

  void set_internal_power_info(std::unique_ptr<LibertyInternalPowerInfo>&& internal_power_info)
  {
    _internal_power_info = std::move(internal_power_info);
  }
  auto& get_internal_power_info() { return _internal_power_info; }

 private:
  std::string _src_port;     //!< The liberty power arc source port
  std::string _snk_port;     //!< The liberty power arc sink port.
  LibertyCell* _owner_cell;  //!< The cell owner the port.

  std::unique_ptr<LibertyInternalPowerInfo> _internal_power_info;  //!< The internal power information.

  DISALLOW_COPY_AND_ASSIGN(LibertyPowerArc);
};

/**
 * @brief The liberty power arc may have the same source and sink port, except
 * the condition is different.
 *
 */
class LibertyPowerArcSet
{
 public:
  LibertyPowerArcSet() = default;
  ~LibertyPowerArcSet() = default;
  LibertyPowerArcSet(LibertyPowerArcSet&& other) noexcept;
  LibertyPowerArcSet& operator=(LibertyPowerArcSet&& rhs) noexcept;

  void addLibertyPowerArc(std::unique_ptr<LibertyPowerArc>&& cell_power_arc) { _power_arcs.emplace_back(std::move(cell_power_arc)); }

  LibertyPowerArc* front() { return _power_arcs.front().get(); }
  auto& get_power_arcs() { return _power_arcs; }

 private:
  Vector<std::unique_ptr<LibertyPowerArc>> _power_arcs;

  DISALLOW_COPY_AND_ASSIGN(LibertyPowerArcSet);
};

/**
 * @brief The macro of foreach power arc, usage:
 * LibertyPowerArcSet* power_arc_set;
 * LibertyPowerArc* power_arc;
 * FOREACH_POWER_ARC(power_arc_set, power_arc)
 * {
 *    do_something_for_power_arc();
 * }
 */
#define FOREACH_POWER_LIB_ARC(power_arc_set, power_arc)                        \
  if (auto& power_arcs = power_arc_set->get_power_arcs(); !power_arcs.empty()) \
    for (auto p = power_arcs.begin(); p != power_arcs.end() ? power_arc = p->get(), true : false; ++p)

/**
 * @brief The timing cell in the liberty.
 *
 */
class LibertyCell : public LibertyObject
{
 public:
  LibertyCell(const char* cell_name, LibertyLibrary* owner_lib);
  ~LibertyCell() override;

  friend LibertyCellPortIterator;
  friend LibertyCellTimingArcSetIterator;
  friend LibertyCellPowerArcSetIterator;

  LibertyCell(LibertyCell&& lib_cell) noexcept;
  LibertyCell& operator=(LibertyCell&& rhs) noexcept;

  const char* get_cell_name() const { return _cell_name.c_str(); }
  auto& get_cell_arcs() { return _cell_arcs; }

  double get_cell_area() const { return _cell_area; }
  void set_cell_area(double cell_area) { _cell_area = cell_area; }

  double get_cell_leakage_power() const { return _cell_leakage_power; }
  void set_cell_leakage_power(double cell_leakage_power) { _cell_leakage_power = cell_leakage_power; }

  std::string get_clock_gating_integrated_cell() const { return _clock_gating_integrated_cell; }
  void set_clock_gating_integrated_cell(std::string clock_gating_integrated_cell)
  {
    _clock_gating_integrated_cell = clock_gating_integrated_cell;
  }

  bool get_is_clock_gating_integrated_cell() const { return _is_clock_gating_integrated_cell; }
  void set_is_clock_gating_integrated_cell(bool is_clock_gating_integrated_cell)
  {
    _is_clock_gating_integrated_cell = is_clock_gating_integrated_cell;
  }

  void addLeakagePower(std::unique_ptr<LibertyLeakagePower>&& leakage_power) { _leakage_power_list.emplace_back(std::move(leakage_power)); }
  auto& get_leakage_power_list() { return _leakage_power_list; }
  std::vector<LibertyLeakagePower*> getLeakagePowerList();
  std::size_t getCellArcSetCount() { return _cell_ports.size(); }

  void addLibertyArc(std::unique_ptr<LibertyArc>&& cell_arc);
  void addLibertyPowerArc(std::unique_ptr<LibertyPowerArc>&& cell_power_arc);
  void addLibertyPort(std::unique_ptr<LibertyPort>&& cell_port)
  {
    _str2ports.emplace(cell_port->get_port_name(), cell_port.get());
    _cell_ports.emplace_back(std::move(cell_port));
  }

  void addLibertyPortBus(std::unique_ptr<LibertyPortBus>&& cell_port_bus)
  {
    _str2portbuses.emplace(cell_port_bus->get_port_name(), cell_port_bus.get());
    _cell_port_buses.emplace_back(std::move(cell_port_bus));
  }

  LibertyPort* get_cell_port_or_port_bus(const char* port_name);
  unsigned get_num_port() { return _cell_ports.size(); }

  auto& get_str2ports() { return _str2ports; }
  auto& get_cell_ports() { return _cell_ports; }

  LibertyLibrary* get_owner_lib() { return _owner_lib; }
  void set_owner_lib(LibertyLibrary* owner_lib) { _owner_lib = owner_lib; }

  std::optional<LibertyArcSet*> findLibertyArcSet(const char* from_port_name, const char* to_port_name, LibertyArc::TimingType timing_type);

  std::optional<LibertyArcSet*> findLibertyArcSet(const char* from_port_name, const char* to_port_name);

  std::vector<LibertyArcSet*> findLibertyArcSet(const char* to_port_name);
  std::optional<LibertyPowerArcSet*> findLibertyPowerArcSet(const char* from_port_name, const char* to_port_name);

  bool hasBufferFunc(LibertyPort* input, LibertyPort* output);
  bool hasInverterFunc(LibertyPort* input, LibertyPort* output);
  void bufferPorts(LibertyPort*& input, LibertyPort*& output);

  bool isBuffer();
  bool isInverter();
  bool isSequentialCell();
  bool isICG();

  void set_is_dont_use() { _is_dont_use = 1; }
  [[nodiscard]] unsigned isDontUse() const { return _is_dont_use; }
  void set_is_macro() { _is_macro_cell = 1; }
  [[nodiscard]] unsigned isMacroCell() const { return _is_macro_cell; }

  double convertTablePowerToMw(double query_table_power);

 private:
  std::string _cell_name;                                                 //!< The liberty cell name.
  double _cell_area;                                                      //!< The liberty cell area.
  double _cell_leakage_power;                                             //!< The cell leakage power of the cell.
  std::string _clock_gating_integrated_cell;                              //!< The clock gate cell.
  bool _is_clock_gating_integrated_cell = false;                          //!< The flag of the clock gate cell.
  std::vector<std::unique_ptr<LibertyLeakagePower>> _leakage_power_list;  //!< All leakage powers of the cell.
  std::vector<std::unique_ptr<LibertyPort>> _cell_ports;
  StrMap<LibertyPort*> _str2ports;  //!< The cell ports.
  std::vector<std::unique_ptr<LibertyPortBus>> _cell_port_buses;
  StrMap<LibertyPortBus*> _str2portbuses;                             //!< The cell port buses.
  std::vector<std::unique_ptr<LibertyArcSet>> _cell_arcs;             //!< All timing arcs of the cell.
  std::vector<std::unique_ptr<LibertyPowerArcSet>> _cell_power_arcs;  //!< All power arcs of the cell.

  LibertyLibrary* _owner_lib;  //!< The owner lib.

  unsigned _is_dont_use : 1 = 0;
  unsigned _is_macro_cell : 1 = 0;
  unsigned _reserved : 30;

  DISALLOW_COPY_AND_ASSIGN(LibertyCell);
};

/**
 * @brief The cell port iterator.
 *
 */
class LibertyCellPortIterator
{
 public:
  explicit LibertyCellPortIterator(LibertyCell* lib_cell);
  ~LibertyCellPortIterator() = default;

  bool hasNext() { return _iter != _lib_cell->_cell_ports.end(); }
  LibertyPort* next() { return _iter++->get(); }

 private:
  LibertyCell* _lib_cell;
  std::vector<std::unique_ptr<LibertyPort>>::iterator _iter;

  DISALLOW_COPY_AND_ASSIGN(LibertyCellPortIterator);
};

/**
 * @brief usage:
 * LibertyCell* lib_cell;
 * LibertyPort* port;
 * FOREACH_CELL_PORT(lib_cell, port)
 * {
 *    do_something_for_port();
 * }
 *
 */
#define FOREACH_CELL_PORT(cell, port) for (ista::LibertyCellPortIterator iter(cell); iter.hasNext() ? port = (iter.next()), true : false;)

/**
 * @brief The macro of foreach leakage power, usage:
 * LibertyCell* lib_cell;
 * LibertyLeakagePower* leakage_power;
 * FOREACH_LEAKAGE_POWER(cell, leakage_power)
 * {
 *    do_something_for_leakage_powers();
 * }
 */
#define FOREACH_LEAKAGE_POWER(cell, leakage_power)                                    \
  if (auto& leakage_powers = cell->get_leakage_power_list(); !leakage_powers.empty()) \
    for (auto p = leakage_powers.begin(); p != leakage_powers.end() ? leakage_power = p->get(), true : false; ++p)

/**
 * @brief The cell timing arc iterator.
 *
 */
class LibertyCellTimingArcSetIterator
{
 public:
  explicit LibertyCellTimingArcSetIterator(LibertyCell* lib_cell);
  ~LibertyCellTimingArcSetIterator() = default;

  bool hasNext() { return _iter != _lib_cell->_cell_arcs.end(); }
  LibertyArcSet* next() { return _iter++->get(); }

 private:
  LibertyCell* _lib_cell;
  std::vector<std::unique_ptr<LibertyArcSet>>::iterator _iter;

  DISALLOW_COPY_AND_ASSIGN(LibertyCellTimingArcSetIterator);
};

/**
 * @brief usage:
 * LibertyCell* lib_cell;
 * LibertyArcSet* timing_arc_set;
 * FOREACH_CELL_TIMING_ARC_SET(lib_cell, timing_arc_set)
 * {
 *    do_something_for_timing_arc_set();
 * }
 *
 */
#define FOREACH_CELL_TIMING_ARC_SET(cell, timing_arc_set) \
  for (ista::LibertyCellTimingArcSetIterator iter(cell); iter.hasNext() ? timing_arc_set = (iter.next()), true : false;)

/**
 * @brief The cell power arc iterator.
 *
 */
class LibertyCellPowerArcSetIterator
{
 public:
  explicit LibertyCellPowerArcSetIterator(LibertyCell* lib_cell);
  ~LibertyCellPowerArcSetIterator() = default;

  bool hasNext() { return _iter != _lib_cell->_cell_power_arcs.end(); }
  LibertyPowerArcSet* next() { return _iter++->get(); }

 private:
  LibertyCell* _lib_cell;

  std::vector<std::unique_ptr<LibertyPowerArcSet>>::iterator _iter;

  DISALLOW_COPY_AND_ASSIGN(LibertyCellPowerArcSetIterator);
};

/**
 * @brief usage:
 * LibertyCell* lib_cell;
 * LibertyPowerArcSet* power_arc_set;
 * FOREACH_POWER_ARC_SET(lib_cell, power_arc_set)
 * {
 *    do_something_for_power_arc_set();
 * }
 */
#define FOREACH_POWER_ARC_SET(cell, power_arc_set) \
  for (ista::LibertyCellPowerArcSetIterator iter(cell); iter.hasNext() ? power_arc_set = (iter.next()), true : false;)

/**
 * @brief The liberty wire load model, such as:
 *
 * wire_load("1K_hvratio_1_4") {
 *   capacitance : 1.774000e-01;
 *   resistance : 3.571429e-03;
 *   slope : 5.000000;
 *   fanout_length( 1, 1.3207 );
 *   fanout_length( 2, 2.9813 );
 *   fanout_length( 3, 5.1135 );
 *   fanout_length( 4, 7.6639 );
 *   fanout_length( 5, 10.0334 );
 *   fanout_length( 6, 12.2296 );
 *   fanout_length( 8, 19.3185 );
 * }
 */
class LibertyWireLoad : public LibertyObject
{
 public:
  explicit LibertyWireLoad(const char* wire_load_name);
  ~LibertyWireLoad() override = default;

  [[nodiscard]] const char* get_wire_load_name() const { return _wire_load_name.c_str(); }

  void set_cap_per_length_unit(double cap_per_length_unit) { _cap_per_length_unit = cap_per_length_unit; }
  auto& get_cap_per_length_unit() { return _cap_per_length_unit; }

  void set_resistance_per_length_unit(double resistance_per_length_unit) { _resistance_per_length_unit = resistance_per_length_unit; }
  auto& get_resistance_per_length_unit() { return _resistance_per_length_unit; }

  void set_slope(double slope) { _slope = slope; }
  auto& get_slope() { return _slope; }

  void add_length_to_map(int fanout, double length) { _fanout_to_length[fanout] = length; }
  auto& get_fanout_to_length() { return _fanout_to_length; }

 private:
  std::string _wire_load_name;                        //!< The wire load name.
  std::map<int, double> _fanout_to_length;            //!< The fanout to length.
  std::optional<double> _cap_per_length_unit;         //!< The cap per length unit.
  std::optional<double> _resistance_per_length_unit;  //!< The resistance per length unit.
  std::optional<double> _slope;                       //!< The slope.
};

/**
 * @brief The liberty lut table template class，such as:
 *  lu_table_template(delay_template_5x5) {
 *  variable_1 : total_output_net_capacitance;
 *  variable_2 : input_net_transition;
 *  index_1 ("1000.0, 1001.0, 1002.0, 1003.0, 1004.0");
 *  index_2 ("1000.0, 1001.0, 1002.0, 1003.0, 1004.0");
 *}
 */
class LibertyLutTableTemplate : public LibertyObject
{
 public:
  enum class Variable
  {
    TOTAL_OUTPUT_NET_CAPACITANCE = 0,
    INPUT_NET_TRANSITION,
    CONSTRAINED_PIN_TRANSITION,
    RELATED_PIN_TRANSITION,
    INPUT_TRANSITION_TIME,
    TIME,
    INPUT_VOLTAGE,
    OUTPUT_VOLTAGE,
    INPUT_NOISE_HEIGHT,
    INPUT_NOISE_WIDTH,
    NORMALIZED_VOLTAGE
  };

  explicit LibertyLutTableTemplate(const char* template_name);
  ~LibertyLutTableTemplate() override = default;

  const char* get_template_name() { return _template_name.c_str(); }

  void set_template_variable1(const char* template_variable1) override
  {
    DLOG_FATAL_IF(!_str2var.contains(template_variable1)) << "not contain the template variable " << template_variable1;
    _template_variable1 = _str2var.at(template_variable1);
  }

  void set_template_variable2(const char* template_variable2) override { _template_variable2 = _str2var.at(template_variable2); }

  void set_template_variable3(const char* template_variable3) override { _template_variable3 = _str2var.at(template_variable3); }

  void set_template_variable4(const char* template_variable4) override { _template_variable4 = _str2var.at(template_variable4); }

  auto get_template_variable1() { return _template_variable1; }
  auto get_template_variable2() { return _template_variable2; }
  auto get_template_variable3() { return _template_variable3; }
  auto get_template_variable4() { return _template_variable4; }

  void addAxis(std::unique_ptr<LibertyAxis>&& axis) override { _axes.push_back(std::move(axis)); }

  auto& get_axes() { return _axes; }

 protected:
  std::string _template_name;

  static const std::map<std::string_view, Variable> _str2var;

  std::optional<Variable> _template_variable1;
  std::optional<Variable> _template_variable2;
  std::optional<Variable> _template_variable3;
  std::optional<Variable> _template_variable4;

  Vector<std::unique_ptr<LibertyAxis>> _axes;  //!< May be zero, one, two, three axes.

  DISALLOW_COPY_AND_ASSIGN(LibertyLutTableTemplate);
};

/**
 * @brief The liberty current template, such as:
 *  output_current_template(ccs_ntin_oload_time_1x1x11) {
 *  variable_1 : input_net_transition ;
 *  variable_2 : total_output_net_capacitance ;
 *  variable_3 : time ;
 *  index_3("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11");
  }
 *
 */
class LibertyCurrentTemplate : public LibertyLutTableTemplate
{
 public:
  explicit LibertyCurrentTemplate(const char* template_name);
  ~LibertyCurrentTemplate() override = default;

  void set_template_axis(std::unique_ptr<LibertyAxis>&& axis) { _template_axis = std::move(axis); }
  LibertyAxis* get_template_axis() { return _template_axis.get(); }

 private:
  std::unique_ptr<LibertyAxis> _template_axis;  //!< The time index template.

  DISALLOW_COPY_AND_ASSIGN(LibertyCurrentTemplate);
};

/**
 * @brief The timing library class.
 *
 */
class LibertyLibrary
{
 public:
  explicit LibertyLibrary(const char* lib_name) : _lib_name(lib_name) {}
  ~LibertyLibrary() = default;

  friend LibertyCellIterator;

  LibertyLibrary(LibertyLibrary&& other) noexcept : _lib_name(std::move(other._lib_name)), _cells(std::move(other._cells)) {}

  LibertyLibrary& operator=(LibertyLibrary&& rhs) noexcept
  {
    _lib_name = std::move(rhs._lib_name);
    _cells = std::move(rhs._cells);

    return *this;
  }

  void addLibertyCell(std::unique_ptr<LibertyCell> lib_cell)
  {
    _str2cell[lib_cell->get_cell_name()] = lib_cell.get();
    _cells.emplace_back(std::move(lib_cell));
  }

  LibertyCell* findCell(const char* cell_name)
  {
    auto p = _str2cell.find(cell_name);
    if (p != _str2cell.end()) {
      return p->second;
    }
    return nullptr;
  }

  void addLutTemplate(std::unique_ptr<LibertyLutTableTemplate> lut_template)
  {
    _str2template[lut_template->get_template_name()] = lut_template.get();
    _lut_templates.emplace_back(std::move(lut_template));
  }

  LibertyLutTableTemplate* getLutTemplate(const char* template_name)
  {
    auto p = _str2template.find(template_name);
    if (p != _str2template.end()) {
      return p->second;
    }
    return nullptr;
  }

  void addLibType(std::unique_ptr<LibertyType> lib_type)
  {
    _str2type[lib_type->get_type_name()] = lib_type.get();
    _types.emplace_back(std::move(lib_type));
  }

  LibertyType* getLibType(const char* lib_type_name)
  {
    auto p = _str2type.find(lib_type_name);
    if (p != _str2type.end()) {
      return p->second;
    }
    return nullptr;
  }

  void addWireLoad(std::unique_ptr<LibertyWireLoad> wire_load)
  {
    _str2wireLoad[wire_load->get_wire_load_name()] = wire_load.get();
    _wire_loads.emplace_back(std::move(wire_load));
  }

  LibertyWireLoad* getWireLoad(const char* wire_load_name)
  {
    auto p = _str2wireLoad.find(wire_load_name);
    if (p != _str2wireLoad.end()) {
      return p->second;
    }
    return nullptr;
  }
  auto get_lib_name() { return _lib_name; }
  auto& get_wire_loads() { return _wire_loads; }

  void set_default_wire_load(const char* wire_load_name) { _default_wire_load = wire_load_name; }

  auto get_default_wire_load() { return _default_wire_load; }

  void set_cap_unit(CapacitiveUnit cap_unit) { _cap_unit = cap_unit; }
  CapacitiveUnit get_cap_unit() { return _cap_unit; }

  void set_resistance_unit(ResistanceUnit resistance_unit) { _resistance_unit = resistance_unit; }
  auto get_resistance_unit() { return _resistance_unit; }

  std::vector<std::unique_ptr<LibertyCell>>& get_cells() { return _cells; }

  void set_default_max_transition(double default_max_transition) { _default_max_transition = default_max_transition; }
  auto& get_default_max_transition() { return _default_max_transition; }

  void set_default_max_fanout(double default_max_fanout) { _default_max_fanout = default_max_fanout; }
  auto& get_default_max_fanout() { return _default_max_fanout; }

  void set_default_fanout_load(double default_fanout_load) { _default_fanout_load = default_fanout_load; }
  auto& get_default_fanout_load() { return _default_fanout_load; }

  void set_nom_voltage(double nom_voltage) { _nom_voltage = nom_voltage; }
  double get_nom_voltage() { return _nom_voltage; }

  void set_slew_lower_threshold_pct_rise(double slew_lower_threshold_pct_rise)
  {
    // change to percent.
    _slew_lower_threshold_pct_rise = slew_lower_threshold_pct_rise / 100.0;
  }
  double get_slew_lower_threshold_pct_rise() { return _slew_lower_threshold_pct_rise; }

  void set_slew_upper_threshold_pct_rise(double slew_upper_threshold_pct_rise)
  {
    // change to percent.
    _slew_upper_threshold_pct_rise = slew_upper_threshold_pct_rise / 100.0;
  }
  double get_slew_upper_threshold_pct_rise() { return _slew_upper_threshold_pct_rise; }

  void set_slew_lower_threshold_pct_fall(double slew_lower_threshold_pct_fall)
  {
    // change to percent.
    _slew_lower_threshold_pct_fall = slew_lower_threshold_pct_fall / 100.0;
  }
  double get_slew_lower_threshold_pct_fall() { return _slew_lower_threshold_pct_fall; }

  void set_slew_upper_threshold_pct_fall(double slew_upper_threshold_pct_fall)
  {
    // change to percent.
    _slew_upper_threshold_pct_fall = slew_upper_threshold_pct_fall / 100.0;
  }
  double get_slew_upper_threshold_pct_fall() { return _slew_upper_threshold_pct_fall; }

  void set_input_threshold_pct_rise(double input_threshold_pct_rise)
  {
    // change to percent.
    _input_threshold_pct_rise = input_threshold_pct_rise / 100.0;
  }
  double get_input_threshold_pct_rise() { return _input_threshold_pct_rise; }

  void set_output_threshold_pct_rise(double output_threshold_pct_rise)
  {
    // change to percent.
    _output_threshold_pct_rise = output_threshold_pct_rise / 100.0;
  }
  double get_output_threshold_pct_rise() { return _output_threshold_pct_rise; }

  void set_input_threshold_pct_fall(double input_threshold_pct_fall) { _input_threshold_pct_fall = input_threshold_pct_fall / 100.0; }
  double get_input_threshold_pct_fall() { return _input_threshold_pct_fall; }

  void set_output_threshold_pct_fall(double output_threshold_pct_fall)
  {
    // change to percent.
    _output_threshold_pct_fall = output_threshold_pct_fall / 100.0;
  }
  double get_output_threshold_pct_fall() { return _output_threshold_pct_fall; }

  void set_slew_derate_from_library(double slew_derate_from_library) { _slew_derate_from_library = slew_derate_from_library; }
  double get_slew_derate_from_library() { return _slew_derate_from_library; }

 private:
  std::string _lib_name;
  std::vector<std::unique_ptr<LibertyCell>> _cells;  //!< The liberty cell, perserve the cell read order.
  StrMap<LibertyCell*> _str2cell;

  Vector<std::unique_ptr<LibertyLutTableTemplate>> _lut_templates;  //!< The timing table lut template, preserve the
                                                                    //!< template order.

  StrMap<LibertyLutTableTemplate*> _str2template;

  Vector<std::unique_ptr<LibertyWireLoad>> _wire_loads;  //!< The wire load models.
  StrMap<LibertyWireLoad*> _str2wireLoad;

  Vector<std::unique_ptr<LibertyType>> _types;  //!< The lib type

  StrMap<LibertyType*> _str2type;

  CapacitiveUnit _cap_unit = CapacitiveUnit::kFF;
  ResistanceUnit _resistance_unit = ResistanceUnit::kkOHM;

  std::optional<double> _default_max_transition;
  std::optional<double> _default_max_fanout;
  std::optional<double> _default_fanout_load;

  std::string _default_wire_load;

  double _nom_voltage = 0.0;  //!< The library nominal voltage

  /// @brief slew threshold
  double _slew_lower_threshold_pct_rise = 0.3;
  double _slew_upper_threshold_pct_rise = 0.7;

  double _slew_lower_threshold_pct_fall = 0.3;
  double _slew_upper_threshold_pct_fall = 0.7;

  /// @brief delay threshold
  double _input_threshold_pct_rise = 0.5;
  double _output_threshold_pct_rise = 0.5;

  double _input_threshold_pct_fall = 0.5;
  double _output_threshold_pct_fall = 0.5;

  // specify how the transition times found in
  // the library need to be derated to match the transition times between the
  // characterization trip points.
  double _slew_derate_from_library = 1.0;

  DISALLOW_COPY_AND_ASSIGN(LibertyLibrary);
};

/**
 * @brief The cell of liberty iterator.
 *
 */
class LibertyCellIterator
{
 public:
  explicit LibertyCellIterator(LibertyLibrary* lib);
  ~LibertyCellIterator() = default;

  bool hasNext();
  LibertyCell* next();

 private:
  LibertyLibrary* _lib;
  std::vector<std::unique_ptr<LibertyCell>>::iterator _iter;

  DISALLOW_COPY_AND_ASSIGN(LibertyCellIterator);
};

/**
 * @brief usage:
 * LibertyLibrary* lib;
 * LibertyCell* cell;
 * FOREACH_LIB_CELL(lib, cell)
 * {
 *    do_something_for_cell();
 * }
 *
 */
#define FOREACH_LIB_CELL(lib, cell) for (ista::LibertyCellIterator iter(lib); iter.hasNext() ? cell = (iter.next()), true : false;)

/**
 * @brief The base class for liberty syntax statement.
 *
 */
class LibertyStmt
{
 public:
  LibertyStmt(const char* file_name, unsigned line_no);
  virtual ~LibertyStmt() = default;

  LibertyStmt(LibertyStmt&& other) noexcept = default;
  LibertyStmt& operator=(LibertyStmt&& rhs) noexcept = default;

  virtual unsigned isSimpleAttrStmt() { return 0; }
  virtual unsigned isComplexAttrStmt() { return 0; }
  virtual unsigned isAttributeStmt() { return 0; }
  virtual unsigned isGroupStmt() { return 0; }

  const char* get_file_name() { return _file_name.c_str(); }
  [[nodiscard]] unsigned get_line_no() const { return _line_no; }

 private:
  std::string _file_name;
  unsigned _line_no = 0;

  DISALLOW_COPY_AND_ASSIGN(LibertyStmt);
};

/**
 * @brief The base class of liberty attribute value.
 * It would be string or float.
 *
 */
class LibertyAttrValue
{
 public:
  LibertyAttrValue() = default;
  virtual ~LibertyAttrValue() = default;

  virtual unsigned isString() { return 0; }
  virtual unsigned isFloat() { return 0; }

  virtual double getFloatValue()
  {
    DLOG_FATAL << "This is unknown value.";
    return 0.0;
  }
  virtual const char* getStringValue()
  {
    DLOG_FATAL << "This is unknown value.";
    return nullptr;
  }
};

/**
 * @brief The liberty float value.
 *
 */
class LibertyFloatValue : public LibertyAttrValue
{
 public:
  explicit LibertyFloatValue(double val) : LibertyAttrValue(), _val(val) {}
  ~LibertyFloatValue() override = default;

  unsigned isFloat() override { return 1; }
  double getFloatValue() override { return _val; }

 private:
  double _val;
};

/**
 * @brief The liberty string value.
 *
 */
class LibertyStringValue : public LibertyAttrValue
{
 public:
  explicit LibertyStringValue(const char* val) : _val(val) {}
  ~LibertyStringValue() override = default;

  unsigned isString() override { return 1; }
  const char* getStringValue() override { return _val.c_str(); }

 private:
  std::string _val;
};

/**
 * @brief The simple attribute statement.
 * For example, drive_strength      : 1;
 */
class LibertySimpleAttrStmt : public LibertyStmt
{
 public:
  explicit LibertySimpleAttrStmt(const char* attri_name, const char* file_name, unsigned line_no)
      : LibertyStmt(file_name, line_no), _attri_name(attri_name), _attribute_value(nullptr)
  {
  }
  ~LibertySimpleAttrStmt() override = default;

  LibertySimpleAttrStmt(LibertySimpleAttrStmt&& other) noexcept = default;

  LibertySimpleAttrStmt& operator=(LibertySimpleAttrStmt&& rhs) noexcept = default;

  unsigned isSimpleAttrStmt() override { return 1; }
  unsigned isAttributeStmt() override { return 1; }

  const char* get_attri_name() { return _attri_name.c_str(); }

  void set_attribute_value(std::unique_ptr<LibertyAttrValue>&& attribute_value) { _attribute_value = std::move(attribute_value); }
  LibertyAttrValue* get_attribute_value() { return _attribute_value.get(); }

 private:
  std::string _attri_name;
  std::unique_ptr<LibertyAttrValue> _attribute_value;

  DISALLOW_COPY_AND_ASSIGN(LibertySimpleAttrStmt);
};

/**
 * @brief The complex attribute statement.
 * For example, index_1
 * ("0.000932129,0.00354597,0.0127211,0.0302424,0.0575396,0.0958408,0.146240");
 */
class LibertyComplexAttrStmt : public LibertyStmt
{
 public:
  explicit LibertyComplexAttrStmt(const char* attri_name, const char* file_name, unsigned line_no)
      : LibertyStmt(file_name, line_no), _attri_name(attri_name)
  {
  }
  ~LibertyComplexAttrStmt() override = default;

  LibertyComplexAttrStmt(LibertyComplexAttrStmt&& other) = default;

  LibertyComplexAttrStmt& operator=(LibertyComplexAttrStmt&& rhs) noexcept = default;

  unsigned isComplexAttrStmt() override { return 1; }
  unsigned isAttributeStmt() override { return 1; }

  const char* get_attri_name() { return _attri_name.c_str(); }

  void set_attribute_values(std::vector<std::unique_ptr<LibertyAttrValue>>&& attri_values) { _attri_values = std::move(attri_values); }
  auto& get_attribute_values() { return _attri_values; }

 private:
  std::string _attri_name;
  std::vector<std::unique_ptr<LibertyAttrValue>> _attri_values;

  DISALLOW_COPY_AND_ASSIGN(LibertyComplexAttrStmt);
};

/**
 * @brief The liberty group statement.
 * For example
 * cell (AND2_X1) {
 * drive_strength       : 1;
 * area                 : 1.064000;
 *  ...
 *  }
 */
class LibertyGroupStmt : public LibertyStmt
{
 public:
  LibertyGroupStmt(const char* group_name, const char* file_name, unsigned line_no)
      : LibertyStmt(file_name, line_no), _group_name(group_name)
  {
  }
  ~LibertyGroupStmt() override = default;

  LibertyGroupStmt(LibertyGroupStmt&& other) noexcept = default;

  LibertyGroupStmt& operator=(LibertyGroupStmt&& rhs) noexcept = default;

  unsigned isGroupStmt() override { return 1; }

  const char* get_group_name() { return _group_name.c_str(); }

  std::vector<std::unique_ptr<LibertyStmt>>& get_stmts() { return _stmts; }
  void set_stmts(std::vector<std::unique_ptr<LibertyStmt>>&& stmts) { _stmts = std::move(stmts); }

  std::vector<std::unique_ptr<LibertyAttrValue>>& get_attri_values() { return _attri_values; }
  void set_attri_values(std::vector<std::unique_ptr<LibertyAttrValue>>&& attri_values) { _attri_values = std::move(attri_values); }

 private:
  std::string _group_name;
  std::vector<std::unique_ptr<LibertyAttrValue>> _attri_values;
  std::vector<std::unique_ptr<LibertyStmt>> _stmts;

  DISALLOW_COPY_AND_ASSIGN(LibertyGroupStmt);
};

/**
 * @brief The library builder from the liberty to the sta analysis lib.
 *
 */
class LibertyBuilder
{
 public:
  enum class LibertyOwnPortType
  {
    kTimingArc = 1,
    kPowerArc = 2
  };
  enum class LibertyOwnPgOrWhenType
  {
    kLibertyLeakagePower = 1,
    kPowerArc = 2
  };
  explicit LibertyBuilder(const char* lib_name) : _lib(std::make_unique<LibertyLibrary>(lib_name)) {}
  ~LibertyBuilder() = default;

  LibertyLibrary* get_lib() { return _lib.get(); }
  std::unique_ptr<LibertyLibrary> takeLib() { return std::move(_lib); }

  void set_obj(LibertyObject* obj) { _obj = obj; }
  LibertyObject* get_obj() { return _obj; }

  void set_cell(LibertyCell* cell) { _obj = _cell = cell; }
  LibertyCell* get_cell() { return _cell; }

  void set_leakage_power(LibertyLeakagePower* leakage_power) { _obj = _leakage_power = leakage_power; }
  LibertyLeakagePower* get_leakage_power() { return _leakage_power; }

  void set_port(LibertyPort* port) { _obj = _port = port; }
  LibertyPort* get_port() { return _port; }

  void set_port_bus(LibertyPortBus* port) { _obj = _port = _port_bus = port; }
  LibertyPortBus* get_port_bus() { return _port_bus; }

  void set_arc(LibertyArc* arc) { _obj = _arc = arc; }
  LibertyArc* get_arc() { return _arc; }

  void set_power_arc(LibertyPowerArc* power_arc) { _obj = _power_arc = power_arc; }
  LibertyPowerArc* get_power_arc() { return _power_arc; }

  void set_table_model(LibertyTableModel* table_model) { _obj = _table_model = table_model; }
  LibertyTableModel* get_table_model() { return _table_model; }

  void set_table(LibertyTable* table) { _obj = _table = table; }
  LibertyTable* get_table() { return _table; }

  void set_current_table(LibertyCCSTable* table)
  {
    //@note pay attention to the obj and current table is not the same.
    // for we need get current table every time we parsed the vector.
    _current_table = table;
  }
  LibertyCCSTable* get_current_table() { return _current_table; }
  void set_own_port_type(LibertyOwnPortType own_port_type) { _own_port_type = own_port_type; }
  LibertyOwnPortType get_own_port_type() { return _own_port_type; }

  void set_own_pg_or_when_type(LibertyOwnPgOrWhenType own_pg_or_when_type) { _own_pg_or_when_type = own_pg_or_when_type; }
  LibertyOwnPgOrWhenType get_own_pg_or_when_type() { return _own_pg_or_when_type; }

 private:
  std::unique_ptr<LibertyLibrary> _lib;  //!< The current lib.

  LibertyObject* _obj = nullptr;                  //< The current library obj except the object below.
  LibertyCell* _cell = nullptr;                   //!< The parsed cell.
  LibertyLeakagePower* _leakage_power = nullptr;  //!< The parsed leakage power.
  LibertyPort* _port = nullptr;                   //!< The parsed port.
  LibertyPortBus* _port_bus = nullptr;            //!< The parsed port bus.
  LibertyArc* _arc = nullptr;                     //!< The parsed timing arc.
  LibertyPowerArc* _power_arc = nullptr;          //!< The parsed power arc.
  LibertyTableModel* _table_model = nullptr;      //!< The parsed table model.
  LibertyTable* _table = nullptr;                 //!< The parsed table.
  LibertyCCSTable* _current_table = nullptr;      //!< The parsed current table.
  LibertyOwnPortType _own_port_type;              //!< The flag of port own timing arc or power arc.
  LibertyOwnPgOrWhenType _own_pg_or_when_type;    //!< The flag of pg port/when own leakage power
                                                  //!< or power arc.
  DISALLOW_COPY_AND_ASSIGN(LibertyBuilder);
};

/**
 * @brief The liberty reader is used to read the related keyword.
 *
 */
class LibertyReader
{
 public:
  explicit LibertyReader(const char* file_name) : _file_name(file_name) {}
  ~LibertyReader() = default;

  LibertyReader(LibertyReader&& other) noexcept = default;
  LibertyReader& operator=(LibertyReader&& rhs) noexcept = default;

  void parseBegin(FILE* fp);
  int parse();
  void parseEnd(FILE* fp);

  unsigned readLib();

  const char* get_file_name() { return _file_name.c_str(); }
  void incrLineNo() { ++_line_no; }
  [[nodiscard]] int get_line_no() const { return _line_no; }

  void clearRecordStr() { _string_buf.erase(); }
  const char* get_record_str() { return _string_buf.c_str(); }
  void recordStr(const char* str) { _string_buf += str; }

  void set_library_group(LibertyGroupStmt* library_group) { _library_group.reset(library_group); }

  char* stringCopy(const char* str);
  void stringDelete(const char* str) { delete[] str; }

  unsigned visitVector(LibertyStmt* group);
  unsigned visitPowerTable(LibertyStmt* group);
  unsigned visitCurrentTable(LibertyStmt* group);
  unsigned visitTable(LibertyStmt* group);
  unsigned visitInternalPower(LibertyStmt* group);
  unsigned visitTiming(LibertyStmt* group);
  unsigned visitPin(LibertyStmt* group);
  unsigned visitBus(LibertyStmt* group);
  unsigned visitLeakagePower(LibertyStmt* group);
  unsigned visitCell(LibertyStmt* group);
  unsigned visitWireLoad(LibertyStmt* group);
  unsigned visitLuTableTemplate(LibertyStmt* group);
  unsigned visitType(LibertyStmt* group);
  unsigned visitOutputCurrentTemplate(LibertyStmt* group);
  unsigned visitLibrary(LibertyStmt* group);
  unsigned visitGroup(LibertyStmt* group);
  unsigned visitSimpleAttri(LibertyStmt* attri);
  unsigned visitAxisOrValues(LibertyStmt* attri);
  unsigned visitComplexAttri(LibertyStmt* attri);

  LibertyGroupStmt* get_library_group() { return _library_group.get(); }
  auto takeLibraryGroup() { return std::move(_library_group); }
  void set_library_builder(std::unique_ptr<LibertyBuilder>&& library_builder) { _library_builder = std::move(library_builder); }
  LibertyBuilder* get_library_builder() { return _library_builder.get(); }

 private:
  std::unique_ptr<LibertyGroupStmt> _library_group;
  std::unique_ptr<LibertyBuilder> _library_builder;

  std::string _file_name;    //!< The verilog file name.
  int _line_no = 0;          //!< The verilog file line no.
  std::string _string_buf;   //!< For flex record inner string.
  void* _scanner = nullptr;  //!< The flex scanner.

  DISALLOW_COPY_AND_ASSIGN(LibertyReader);
};

/**
 * @brief This is the top interface class for liberty module.
 *
 */
class Liberty
{
 public:
  Liberty() = default;
  ~Liberty() = default;

  std::unique_ptr<LibertyLibrary> loadLiberty(const char* file_name);

 private:
  DISALLOW_COPY_AND_ASSIGN(Liberty);
};

}  // namespace ista
