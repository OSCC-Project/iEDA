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
 * @file Lib.h
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
#include "BTreeMap.hh"
#include "FlatMap.hh"
#include "FlatSet.hh"
#include "LibParserRustC.hh"
#include "Vector.hh"
#include "include/Config.hh"
#include "include/Type.hh"
#include "log/Log.hh"
#include "string/Str.hh"
#include "string/StrMap.hh"

namespace ista {

class LibType;
class LibCell;
class LibLibrary;
class LibAttrValue;
class LibAxis;
class LibLutTableTemplate;
class LibVectorTable;
class LibertyExpr;

/**
 * @brief The base object of the library.
 *
 */
class LibObject
{
 public:
  LibObject() = default;
  virtual ~LibObject() = default;

  virtual void addAxis(std::unique_ptr<LibAxis>&& axis) { LOG_FATAL << "not support"; }
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

  FORBIDDEN_COPY(LibObject);
};

/**
 * @brief The liberty axis, which the lut table consist of.
 *
 */
class LibAxis : public LibObject
{
 public:
  explicit LibAxis(const char* axis_name);
  ~LibAxis() override = default;

  LibAxis(LibAxis&& other) noexcept;
  LibAxis& operator=(LibAxis&& rhs) noexcept;

  const char* get_axis_name() { return _axis_name.c_str(); }

  void set_axis_values(std::vector<std::unique_ptr<LibAttrValue>>&& table_values) { _axis_values = std::move(table_values); }

  auto& get_axis_values() { return _axis_values; }
  std::size_t get_axis_size() { return _axis_values.size(); }

  double operator[](std::size_t index);

 private:
  std::string _axis_name;  //!< The axis name.

  std::vector<std::unique_ptr<LibAttrValue>> _axis_values;  //!< The axis sample values.

  FORBIDDEN_COPY(LibAxis);
};

/**
 * @brief The liberty NLDM table.
 *
 */
class LibTable : public LibObject
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
    kFallPower = 9,
    // sigma
    kCellRiseSigma = 10,
    kCellFallSigma = 12,
    kRiseTransitionSigma = 14,
    kFallTransitionSigma = 16
  };

  enum class CornerType : int
  {
    kDefault = 0,
    kEarly = 1,
    kLate = 2
  };

  static const std::map<std::string, TableType> _str2TableType;
  static const unsigned _time_index = 2;

  LibTable(TableType table_type, LibLutTableTemplate* table_template);

  ~LibTable() override = default;

  LibTable(LibTable&& other) noexcept;
  LibTable& operator=(LibTable&& rhs) noexcept;

  void addAxis(std::unique_ptr<LibAxis>&& axis) override { _axes.push_back(std::move(axis)); }

  LibAxis& getAxis(unsigned int index);

  Vector<std::unique_ptr<LibAxis>>& get_axes();
  auto getAxesSize() { return _axes.size(); }

  void set_table_values(std::vector<std::unique_ptr<LibAttrValue>>&& table_values) { _table_values = std::move(table_values); }
  auto& get_table_values() { return _table_values; }

  void set_corner_type(CornerType corner_type) { _corner_type = corner_type; }
  CornerType get_corner_type() { return _corner_type; }

  TableType get_table_type() { return _table_type; }

  void set_table_template(LibLutTableTemplate* table_template) { _table_template = table_template; }
  LibLutTableTemplate* get_table_template() { return _table_template; }

  double findValue(double slew, double constrain_slew_or_load);

  double driveResistance();

 private:
  Vector<std::unique_ptr<LibAxis>> _axes;                    //!< May be zero, one, two, three axes.
  std::vector<std::unique_ptr<LibAttrValue>> _table_values;  //!< The axis values.
  TableType _table_type;                                     //!< The table type.

  CornerType _corner_type = CornerType::kDefault;
  LibLutTableTemplate* _table_template;  //!< The lut template.

  FORBIDDEN_COPY(LibTable);
};

/**
 * @brief The CCS model simulation information.
 *
 */
struct LibCurrentSimuInfo
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
class LibVectorTable : public LibTable
{
 public:
  LibVectorTable(TableType table_type, LibLutTableTemplate* table_template);
  ~LibVectorTable() override = default;

  LibVectorTable(LibVectorTable&& other) noexcept;
  LibVectorTable& operator=(LibVectorTable&& rhs) noexcept;

  void set_ref_time(double ref_time) { _ref_time = ref_time; }
  [[nodiscard]] double get_ref_time() const { return _ref_time; }

  std::tuple<double, int> getSimulationTotalTimeAndNumPoints();

  std::vector<double> getOutputCurrent(std::optional<LibCurrentSimuInfo>& simu_info);

 private:
  double _ref_time = 0.0;  //!< The current reference time.

  FORBIDDEN_COPY(LibVectorTable);
};

/**
 * @brief The output current data for upper layer interface.
 *
 */
class LibCurrentData
{
 public:
  LibCurrentData(LibVectorTable* low_low, LibVectorTable* low_high, LibVectorTable* high_low, LibVectorTable* high_high, double slew,
                 double load);
  ~LibCurrentData() = default;

  LibCurrentData(const LibCurrentData& orig) = default;
  LibCurrentData& operator=(const LibCurrentData& rhs) = default;

  LibCurrentData(LibCurrentData&& orig) = default;
  LibCurrentData& operator=(LibCurrentData&& rhs) = default;

  LibCurrentData* copy() { return new LibCurrentData(*this); }

  LibVectorTable* get_low_low() { return _low_low; }
  LibVectorTable* get_low_high() { return _low_high; }
  LibVectorTable* get_high_low() { return _high_low; }
  LibVectorTable* get_high_high() { return _high_high; }

  std::tuple<double, int> getSimulationTotalTimeAndNumPoints();

  std::vector<double> getOutputCurrent(std::optional<LibCurrentSimuInfo>& simu_info);

 private:
  LibVectorTable* _low_low;    //!< low slew and low load
  LibVectorTable* _low_high;   //!< low slew and high load
  LibVectorTable* _high_low;   //!< high slew and low load
  LibVectorTable* _high_high;  //!< high slew and high load

  double _slew;
  double _load;
};

/**
 * @brief The liberty CCS table include one or more vector table.
 *
 */
class LibCCSTable : public LibObject
{
 public:
  explicit LibCCSTable(LibTable::TableType table_type);
  ~LibCCSTable() override = default;
  void addTable(std::unique_ptr<LibVectorTable>&& current_table) { _vector_tables.emplace_back(std::move(current_table)); }
  auto get_table_type() { return _table_type; }
  auto& get_vector_tables() { return _vector_tables; }

 private:
  LibTable::TableType _table_type;                              //!< The table type.
  std::vector<std::unique_ptr<LibVectorTable>> _vector_tables;  //!< The current tables.

  FORBIDDEN_COPY(LibCCSTable);
};

#define STR_TO_TABLE_TYPE(str) LibTable::_str2TableType.at(str)

/**
 * @brief The liberty table model, include delay model and check model.
 *
 */
class LibTableModel : public LibObject
{
 public:
  LibTableModel() = default;
  ~LibTableModel() override = default;
  virtual unsigned isDelayModel() { return 0; }
  virtual unsigned isCheckModel() { return 0; }
  virtual unsigned isPowerModel() { return 0; }
  virtual unsigned addTable(std::unique_ptr<LibTable>&& table) = 0;
  virtual LibTable* getTable(int index) = 0;
  virtual std::optional<double> gateDelay(TransType trans_type, double slew, double load)
  {
    LOG_FATAL << "not support";
    return 0.0;
  }
  virtual std::optional<double> gateDelaySigma(AnalysisMode mode, TransType trans_type, double slew, double load) { return 0.0; }
  virtual std::optional<double> gateSlew(TransType trans_type, double slew, double load)
  {
    LOG_FATAL << "not support";
    return 0.0;
  }
  virtual std::optional<double> gateSlewSigma(AnalysisMode mode, TransType trans_type, double slew, double load) { return 0.0; }
  virtual std::optional<double> gateCheckConstrain(TransType trans_type, double slew, double load)
  {
    LOG_FATAL << "not support";
    return 0.0;
  }

  virtual std::unique_ptr<LibCurrentData> gateOutputCurrent(TransType trans_type, double slew, double load)
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
  FORBIDDEN_COPY(LibTableModel);
};

#define CAST_TYPE_TO_INDEX(type) ((static_cast<int>(type) > 3) ? (static_cast<int>(type) - 4) : static_cast<int>(type))
#define CAST_CURRENT_TYPE_TO_INDEX(type) (static_cast<int>(type) - 6)
#define CAST_POWER_TYPE_TO_INDEX(type) (static_cast<int>(type) - 8)

/**
 * @brief The liberty delay model.
 *
 */
class LibDelayTableModel final : public LibTableModel
{
 public:
  static constexpr size_t kTableNum = 4;         //!< The model contain delay/slew, rise/fall four table, eight sigma
                                                 //!< table(rise/fall, early/late, delay/slew ).
  static constexpr size_t kCurrentTableNum = 2;  //!< Current rise/fall table.

  unsigned isDelayModel() override { return 1; }

  LibDelayTableModel() = default;
  ~LibDelayTableModel() = default;

  LibDelayTableModel(LibDelayTableModel&& other) noexcept;
  LibDelayTableModel& operator=(LibDelayTableModel&& rhs) noexcept;

  int calcShiftIndex(LibTable::CornerType corner_type)
  {
    int shift = 0;
    if (corner_type != LibTable::CornerType::kDefault) {
      shift = -2 + (static_cast<int>(corner_type) - 1);
    }
    return shift;
  }

  unsigned addTable(std::unique_ptr<LibTable>&& table)
  {
    auto table_type = table->get_table_type();
    int index = CAST_TYPE_TO_INDEX(table_type);
    int shift_index = calcShiftIndex(table->get_corner_type());
    index += shift_index;

    _tables.at(index) = std::move(table);
    return 1;
  }

  LibTable* getTable(int index) override { return _tables[index].get(); }

  unsigned addCurrentTable(std::unique_ptr<LibCCSTable>&& table)
  {
    auto table_type = table->get_table_type();
    _current_tables[CAST_CURRENT_TYPE_TO_INDEX(table_type)] = std::move(table);
    return 1;
  }

  std::optional<double> gateDelay(TransType trans_type, double slew, double load) override;
  std::optional<double> gateDelaySigma(AnalysisMode mode, TransType trans_type, double slew, double load) override;
  std::optional<double> gateSlew(TransType trans_type, double slew, double load) override;
  std::optional<double> gateSlewSigma(AnalysisMode mode, TransType trans_type, double slew, double load) override;
  std::unique_ptr<LibCurrentData> gateOutputCurrent(TransType trans_type, double slew, double load) override;

  double driveResistance() override;

 private:
  std::array<std::unique_ptr<LibTable>, kTableNum> _tables;  // NLDM table,include cell rise/cell fall/rise transition/fall transition,
                                                             // and  eight sigma table(rise/fall, early/late, delay/slew ).
  std::array<std::unique_ptr<LibCCSTable>,
             kCurrentTableNum>  // Output current rise/fall.
      _current_tables;

  FORBIDDEN_COPY(LibDelayTableModel);
};

/**
 * @brief The liberty check model.
 *
 */
class LibCheckTableModel final : public LibTableModel
{
 public:
  static constexpr size_t kTableNum = 2;  //!< The model contain rise/fall constrain two tables.

  unsigned isCheckModel() override { return 1; }

  LibCheckTableModel() = default;
  ~LibCheckTableModel() override = default;

  LibCheckTableModel(LibCheckTableModel&& other) noexcept;
  LibCheckTableModel& operator=(LibCheckTableModel&& rhs) noexcept;

  unsigned addTable(std::unique_ptr<LibTable>&& table) override
  {
    auto table_type = table->get_table_type();
    _tables[CAST_TYPE_TO_INDEX(table_type)] = std::move(table);
    return 1;
  }

  LibTable* getTable(int index) override { return _tables[index].get(); }
  std::optional<double> gateCheckConstrain(TransType trans_type, double slew, double constrain_slew) override;

 private:
  std::array<std::unique_ptr<LibTable>, kTableNum> _tables;

  FORBIDDEN_COPY(LibCheckTableModel);
};

/**
 * @brief The liberty power model.
 *
 */
class LibPowerTableModel final : public LibTableModel
{
 public:
  static constexpr size_t kTableNum = 2;  //!< The model contain rise/fall power two table.
  unsigned isPowerModel() override { return 1; }

  LibPowerTableModel() = default;
  ~LibPowerTableModel() override = default;

  LibPowerTableModel(LibPowerTableModel&& other) noexcept;
  LibPowerTableModel& operator=(LibPowerTableModel&& rhs) noexcept;

  unsigned addTable(std::unique_ptr<LibTable>&& table)
  {
    auto table_type = table->get_table_type();
    _tables[CAST_POWER_TYPE_TO_INDEX(table_type)] = std::move(table);
    return 1;
  }
  LibTable* getTable(int index) override { return _tables[index].get(); }

  double gatePower(TransType trans_type, double slew, std::optional<double> load) override;

 private:
  std::array<std::unique_ptr<LibTable>, kTableNum> _tables;  // power table,include rise power/fall power.
  FORBIDDEN_COPY(LibPowerTableModel);
};

/**
 * @brief class for internal power information
 *
 */
class LibInternalPowerInfo : public LibObject
{
 public:
  void set_related_pg_port(const char* related_pg_port) { _related_pg_port = related_pg_port; }
  auto& get_related_pg_port() { return _related_pg_port; }

  void set_when(const char* when) { _when = when; }
  auto& get_when() { return _when; }

  void set_power_table_model(std::unique_ptr<LibTableModel>&& power_table_model) { _power_table_model = std::move(power_table_model); }
  LibTableModel* get_power_table_model() { return _power_table_model.get(); }

  double gatePower(TransType trans_type, double slew, std::optional<double> load)
  {
    return _power_table_model->gatePower(trans_type, slew, load);
  }

 private:
  std::string _related_pg_port;                       //!< The liberty power arc related pg port.
  std::string _when;                                  //!< The liberty power arc related pg port.
  std::unique_ptr<LibTableModel> _power_table_model;  //!< The pin power table model.
};

/**
 * @brief The port in the cell.
 *
 */
class LibPort : public LibObject
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

  explicit LibPort(const char* port_name);
  ~LibPort() override = default;

  LibPort(LibPort&& other) noexcept;
  LibPort& operator=(LibPort&& rhs) noexcept;

  const char* get_port_name() { return _port_name.c_str(); }
  void set_ower_cell(LibCell* ower_cell) { _ower_cell = ower_cell; }
  LibCell* get_ower_cell() { return _ower_cell; }

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

  void set_func_expr(RustLibertyExpr* lib_expr) { _func_expr = lib_expr; }
  RustLibertyExpr* get_func_expr() { return _func_expr; }

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

  void addInternalPower(std::unique_ptr<LibInternalPowerInfo>&& internal_power)
  {
    _internal_powers.emplace_back(std::move(internal_power));
  }
  auto& get_internal_powers() { return _internal_powers; }

 private:
  std::string _port_name;
  LibCell* _ower_cell;  //!< The cell owner the port.
  LibertyPortType _port_type = LibertyPortType::kDefault;
  bool _clock_gate_clock_pin = false;   //!< The flag of gate clock pin.
  bool _clock_gate_enable_pin = false;  //!< The flag of gate enable pin.
  RustLibertyExpr* _func_expr = nullptr;
  std::string _func_expr_str;                                        //!< store func expr string for debug.
  double _port_cap = 0.0;                                            //!< The input pin corresponding to the port has capacitance.
  std::array<std::optional<double>, MODE_TRANS_SPLIT> _port_caps{};  //!< May be port cap split max rise, max fall, min rise,
                                                                     //!< min fall.
  std::array<std::optional<double>, MODE_SPLIT> _cap_limits{};
  std::array<std::optional<double>, MODE_SPLIT> _slew_limits{};

  std::optional<double> _fanout_load;

  Vector<std::unique_ptr<LibInternalPowerInfo>> _internal_powers;  //!< The internal power information.

  FORBIDDEN_COPY(LibPort);
};

/**
 * @brief The macro of foreach internal power, usage:
 * LibPort* port;
 * LibInternalPowerInfo* internal_power;
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
class LibType : public LibObject
{
 public:
  explicit LibType(std::string&& type_name) : _type_name(std::move(type_name)) {}

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
class LibPortBus : public LibPort
{
 public:
  explicit LibPortBus(const char* port_bus_name);
  ~LibPortBus() override = default;

  unsigned isLibertyPortBus() override { return 1; }

  void addlibertyPort(std::unique_ptr<LibPort>&& port) { _ports.push_back(std::move(port)); }

  auto getBusSize() { return _bus_type ? _bus_type->get_bit_width() : _ports.size(); }

  void set_bus_type(LibType* bus_type) { _bus_type = bus_type; }
  auto* get_bus_type() { return _bus_type; }

  LibPort* operator[](int index) { return _ports.empty() ? this : _ports[index].get(); }

 private:
  Vector<std::unique_ptr<LibPort>> _ports;  //!< The bus ports.
  LibType* _bus_type = nullptr;

  FORBIDDEN_COPY(LibPortBus);
};

/**
 * @brief The leakage power in the cell.
 *
 */
class LibLeakagePower : public LibObject
{
 public:
  LibLeakagePower();
  ~LibLeakagePower() override = default;

  LibLeakagePower(LibLeakagePower&& other) noexcept;
  LibLeakagePower& operator=(LibLeakagePower&& rhs) noexcept;

  void set_owner_cell(LibCell* ower_cell) { _owner_cell = ower_cell; }
  LibCell* get_owner_cell() { return _owner_cell; }
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
  LibCell* _owner_cell;          //!< The cell owner the port.

  FORBIDDEN_COPY(LibLeakagePower);
};

/**
 * @brief The timing arc in the liberty.
 *
 */
class LibArc : public LibObject
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

  LibArc();
  ~LibArc() override = default;

  LibArc(LibArc&& other) noexcept;
  LibArc& operator=(LibArc&& rhs) noexcept;

  void set_src_port(const char* src_port) { _src_port = src_port; }
  void set_snk_port(const char* snk_port) { _snk_port = snk_port; }
  const char* get_src_port() { return _src_port.c_str(); }
  const char* get_snk_port() { return _snk_port.c_str(); }

  void set_timing_sense(const char* timing_sense);
  TimingSense get_timing_sense() { return _timing_sense; }

  void set_timing_type(const char* timing_type);
  TimingType get_timing_type() { return _timing_type; }
  bool isMatchTimingType(TransType trans_type);

  void set_owner_cell(LibCell* ower_cell) { _owner_cell = ower_cell; }
  LibCell* get_owner_cell() { return _owner_cell; }

  void set_is_disable_arc() { _is_disable_arc = 1;}
  unsigned isDisableArc() { return _is_disable_arc; }

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

  void set_table_model(std::unique_ptr<LibTableModel>&& table_model) { _table_model = std::move(table_model); }
  LibTableModel* get_table_model() { return _table_model.get(); }

  double getDelayOrConstrainCheckNs(TransType trans_type, double slew, double load_or_constrain_slew);
  double getDelaySigma(AnalysisMode mode, TransType trans_type, double slew, double load_or_constrain_slew);
  double getSlewNs(TransType trans_type, double slew, double load);
  double getSlewSigma(AnalysisMode mode, TransType trans_type, double slew, double load);

  std::unique_ptr<LibCurrentData> getOutputCurrent(TransType trans_type, double slew, double load);

  double getDriveResistance() { return _table_model->driveResistance(); }

 private:
  std::string _src_port;                           //!< The liberty timing arc source port, for liberty
                                                   //!< file port may be behind the arc, so we use port
                                                   //!< name, fix me.
  std::string _snk_port;                           //!< The liberty timing arc sink port.
  LibCell* _owner_cell;                            //!< The cell owner the port.
  TimingSense _timing_sense;                       //!< The arc timing sense.
  TimingType _timing_type = TimingType::kDefault;  //!< The arc timing type.

  std::unique_ptr<LibTableModel> _table_model;  //!< The arc timing model.

  static BTreeMap<std::string, TimingType> _str_to_type;

  unsigned _is_disable_arc = 0; //!< Forbidden arc.

  FORBIDDEN_COPY(LibArc);
};

/**
 * @brief The liberty arc may have the same source and sink port, except the
 * condition is different.
 *
 */
class LibArcSet
{
 public:
  LibArcSet() = default;
  ~LibArcSet() = default;
  LibArcSet(LibArcSet&& other) noexcept;
  LibArcSet& operator=(LibArcSet&& rhs) noexcept;

  void addLibertyArc(std::unique_ptr<LibArc>&& cell_arc) { _arcs.emplace_back(std::move(cell_arc)); }

  LibArc* front() { return _arcs.front().get(); }
  auto& get_arcs() { return _arcs; }

 private:
  Vector<std::unique_ptr<LibArc>> _arcs;

  FORBIDDEN_COPY(LibArcSet);
};

/**
 * @brief The power arc in the liberty.
 *
 */
class LibPowerArc : public LibObject
{
 public:
  LibPowerArc();
  ~LibPowerArc() override = default;

  LibPowerArc(LibPowerArc&& other) noexcept;
  LibPowerArc& operator=(LibPowerArc&& rhs) noexcept;

  void set_src_port(const char* src_port) { _src_port = src_port; }
  void set_snk_port(const char* snk_port) { _snk_port = snk_port; }

  const char* get_src_port() { return _src_port.c_str(); }
  const char* get_snk_port() { return _snk_port.c_str(); }

  bool isSrcPortEmpty() { return _src_port.empty(); }
  bool isSnkPortEmpty() { return _snk_port.empty(); }

  void set_owner_cell(LibCell* ower_cell) { _owner_cell = ower_cell; }
  LibCell* get_owner_cell() { return _owner_cell; }

  void set_related_pg_port(const char* related_pg_port) { _internal_power_info->set_related_pg_port(related_pg_port); }
  const char* get_related_pg_port() { return _internal_power_info->get_related_pg_port().c_str(); }

  void set_when(const char* when) { _internal_power_info->set_when(when); }
  std::string get_when() { return _internal_power_info->get_when(); }

  void set_power_table_model(std::unique_ptr<LibTableModel>&& power_table_model)
  {
    _internal_power_info->set_power_table_model(std::move(power_table_model));
  }
  LibTableModel* get_power_table_model() { return _internal_power_info->get_power_table_model(); }

  void set_internal_power_info(std::unique_ptr<LibInternalPowerInfo>&& internal_power_info)
  {
    _internal_power_info = std::move(internal_power_info);
  }
  auto& get_internal_power_info() { return _internal_power_info; }

 private:
  std::string _src_port;  //!< The liberty power arc source port
  std::string _snk_port;  //!< The liberty power arc sink port.
  LibCell* _owner_cell;   //!< The cell owner the port.

  std::unique_ptr<LibInternalPowerInfo> _internal_power_info;  //!< The internal power information.

  FORBIDDEN_COPY(LibPowerArc);
};

/**
 * @brief The liberty power arc may have the same source and sink port, except
 * the condition is different.
 *
 */
class LibPowerArcSet
{
 public:
  LibPowerArcSet() = default;
  ~LibPowerArcSet() = default;
  LibPowerArcSet(LibPowerArcSet&& other) noexcept;
  LibPowerArcSet& operator=(LibPowerArcSet&& rhs) noexcept;

  void addLibertyPowerArc(std::unique_ptr<LibPowerArc>&& cell_power_arc) { _power_arcs.emplace_back(std::move(cell_power_arc)); }

  LibPowerArc* front() { return _power_arcs.front().get(); }
  auto& get_power_arcs() { return _power_arcs; }

 private:
  Vector<std::unique_ptr<LibPowerArc>> _power_arcs;

  FORBIDDEN_COPY(LibPowerArcSet);
};

/**
 * @brief The macro of foreach power arc, usage:
 * LibPowerArcSet* power_arc_set;
 * LibPowerArc* power_arc;
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
class LibCell : public LibObject
{
 public:
  LibCell(const char* cell_name, LibLibrary* owner_lib);
  ~LibCell() override;

  LibCell(LibCell&& lib_cell) noexcept;
  LibCell& operator=(LibCell&& rhs) noexcept;

  const char* get_cell_name() const { return _cell_name.c_str(); }
  auto& get_cell_arcs() { return _cell_arcs; }
  auto& get_cell_power_arcs() { return _cell_power_arcs; }

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

  void addLeakagePower(std::unique_ptr<LibLeakagePower>&& leakage_power) { _leakage_power_list.emplace_back(std::move(leakage_power)); }
  auto& get_leakage_power_list() { return _leakage_power_list; }
  std::vector<LibLeakagePower*> getLeakagePowerList();
  std::size_t getCellArcSetCount() { return _cell_ports.size(); }

  void addLibertyArc(std::unique_ptr<LibArc>&& cell_arc);
  void addLibertyPowerArc(std::unique_ptr<LibPowerArc>&& cell_power_arc);
  void addLibertyPort(std::unique_ptr<LibPort>&& cell_port)
  {
    _str2ports.emplace(cell_port->get_port_name(), cell_port.get());
    _cell_ports.emplace_back(std::move(cell_port));
  }

  void addLibertyPortBus(std::unique_ptr<LibPortBus>&& cell_port_bus)
  {
    _str2portbuses.emplace(cell_port_bus->get_port_name(), cell_port_bus.get());
    _cell_port_buses.emplace_back(std::move(cell_port_bus));
  }

  LibPort* get_cell_port_or_port_bus(const char* port_name);
  unsigned get_num_port() { return _cell_ports.size(); }

  auto& get_str2ports() { return _str2ports; }
  auto& get_cell_ports() { return _cell_ports; }

  LibLibrary* get_owner_lib() { return _owner_lib; }
  void set_owner_lib(LibLibrary* owner_lib) { _owner_lib = owner_lib; }

  std::optional<LibArcSet*> findLibertyArcSet(const char* from_port_name, const char* to_port_name, LibArc::TimingType timing_type);

  std::optional<LibArcSet*> findLibertyArcSet(const char* from_port_name, const char* to_port_name);

  std::vector<LibArcSet*> findLibertyArcSet(const char* to_port_name);
  std::optional<LibPowerArcSet*> findLibertyPowerArcSet(const char* from_port_name, const char* to_port_name);

  bool hasBufferFunc(LibPort* input, LibPort* output);
  bool hasInverterFunc(LibPort* input, LibPort* output);
  void bufferPorts(LibPort*& input, LibPort*& output);

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
  std::string _cell_name;                                             //!< The liberty cell name.
  double _cell_area;                                                  //!< The liberty cell area.
  double _cell_leakage_power;                                         //!< The cell leakage power of the cell.
  std::string _clock_gating_integrated_cell;                          //!< The clock gate cell.
  bool _is_clock_gating_integrated_cell = false;                      //!< The flag of the clock gate cell.
  std::vector<std::unique_ptr<LibLeakagePower>> _leakage_power_list;  //!< All leakage powers of the cell.
  std::vector<std::unique_ptr<LibPort>> _cell_ports;
  StrMap<LibPort*> _str2ports;  //!< The cell ports.
  std::vector<std::unique_ptr<LibPortBus>> _cell_port_buses;
  StrMap<LibPortBus*> _str2portbuses;                             //!< The cell port buses.
  std::vector<std::unique_ptr<LibArcSet>> _cell_arcs;             //!< All timing arcs of the cell.
  std::vector<std::unique_ptr<LibPowerArcSet>> _cell_power_arcs;  //!< All power arcs of the cell.

  LibLibrary* _owner_lib;  //!< The owner lib.

  unsigned _is_dont_use : 1;
  unsigned _is_macro_cell : 1;
  unsigned _reserved : 30;

  FORBIDDEN_COPY(LibCell);
};

/**
 * @brief usage:
 * LibCell* lib_cell;
 * LibPort* port;
 * FOREACH_CELL_PORT(lib_cell, port)
 * {
 *    do_something_for_port();
 * }
 *
 */
#define FOREACH_CELL_PORT(cell, port)                                                               \
  for (std::vector<std::unique_ptr<ista::LibPort>>::iterator iter = cell->get_cell_ports().begin(); \
       (iter != cell->get_cell_ports().end()) ? port = (iter++->get()), true : false;)

/**
 * @brief The macro of foreach leakage power, usage:
 * LibCell* lib_cell;
 * LibLeakagePower* leakage_power;
 * FOREACH_LEAKAGE_POWER(cell, leakage_power)
 * {
 *    do_something_for_leakage_powers();
 * }
 */
#define FOREACH_LEAKAGE_POWER(cell, leakage_power)                                    \
  if (auto& leakage_powers = cell->get_leakage_power_list(); !leakage_powers.empty()) \
    for (auto p = leakage_powers.begin(); p != leakage_powers.end() ? leakage_power = p->get(), true : false; ++p)

/**
 * @brief usage:
 * LibCell* lib_cell;
 * LibArcSet* timing_arc_set;
 * FOREACH_CELL_TIMING_ARC_SET(lib_cell, timing_arc_set)
 * {
 *    do_something_for_timing_arc_set();
 * }
 *
 */
#define FOREACH_CELL_TIMING_ARC_SET(cell, timing_arc_set)                                      \
  for (std::vector<std::unique_ptr<LibArcSet>>::iterator iter = cell->get_cell_arcs().begin(); \
       iter != cell->get_cell_arcs().end() ? timing_arc_set = iter++->get(), true : false;)

/**
 * @brief usage:
 * LibCell* lib_cell;
 * LibPowerArcSet* power_arc_set;
 * FOREACH_POWER_ARC_SET(lib_cell, power_arc_set)
 * {
 *    do_something_for_power_arc_set();
 * }
 */
#define FOREACH_POWER_ARC_SET(cell, power_arc_set)                                                              \
  for (std::vector<std::unique_ptr<ista::LibPowerArcSet>>::iterator iter = cell->get_cell_power_arcs().begin(); \
       iter != cell->get_cell_power_arcs().end() ? power_arc_set = iter++->get(), true : false;)

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
class LibWireLoad : public LibObject
{
 public:
  explicit LibWireLoad(const char* wire_load_name);
  ~LibWireLoad() override = default;

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
 *  variable_1 : input_net_transition;
 *  variable_2 : total_output_net_capacitance;
 *  index_1 ("1000.0, 1001.0, 1002.0, 1003.0, 1004.0");
 *  index_2 ("1000.0, 1001.0, 1002.0, 1003.0, 1004.0");
 *}
 */
class LibLutTableTemplate : public LibObject
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

  explicit LibLutTableTemplate(const char* template_name);
  ~LibLutTableTemplate() override = default;

  const char* get_template_name() { return _template_name.c_str(); }

  void set_template_variable1(const char* template_variable1) override
  {
    DLOG_FATAL_IF(_str2var.find(template_variable1) == _str2var.end()) << "not contain the template variable " << template_variable1;
    _template_variable1 = _str2var.at(template_variable1);
  }

  void set_template_variable2(const char* template_variable2) override { _template_variable2 = _str2var.at(template_variable2); }

  void set_template_variable3(const char* template_variable3) override { _template_variable3 = _str2var.at(template_variable3); }

  void set_template_variable4(const char* template_variable4) override { _template_variable4 = _str2var.at(template_variable4); }

  auto get_template_variable1() { return _template_variable1; }
  auto get_template_variable2() { return _template_variable2; }
  auto get_template_variable3() { return _template_variable3; }
  auto get_template_variable4() { return _template_variable4; }

  void addAxis(std::unique_ptr<LibAxis>&& axis) override { _axes.push_back(std::move(axis)); }

  auto& get_axes() { return _axes; }

 protected:
  std::string _template_name;

  static const std::map<std::string_view, Variable> _str2var;

  std::optional<Variable> _template_variable1;
  std::optional<Variable> _template_variable2;
  std::optional<Variable> _template_variable3;
  std::optional<Variable> _template_variable4;

  Vector<std::unique_ptr<LibAxis>> _axes;  //!< May be zero, one, two, three axes.

  FORBIDDEN_COPY(LibLutTableTemplate);
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
class LibCurrentTemplate : public LibLutTableTemplate
{
 public:
  explicit LibCurrentTemplate(const char* template_name);
  ~LibCurrentTemplate() override = default;

  void set_template_axis(std::unique_ptr<LibAxis>&& axis) { _template_axis = std::move(axis); }
  LibAxis* get_template_axis() { return _template_axis.get(); }

 private:
  std::unique_ptr<LibAxis> _template_axis;  //!< The time index template.

  FORBIDDEN_COPY(LibCurrentTemplate);
};

/**
 * @brief The timing library class.
 *
 */
class LibLibrary : public LibObject
{
 public:
  explicit LibLibrary(const char* lib_name) : _lib_name(lib_name) {}
  ~LibLibrary() = default;

  LibLibrary(LibLibrary&& other) noexcept : _lib_name(std::move(other._lib_name)), _cells(std::move(other._cells)) {}

  LibLibrary& operator=(LibLibrary&& rhs) noexcept
  {
    _lib_name = std::move(rhs._lib_name);
    _cells = std::move(rhs._cells);

    return *this;
  }

  void addLibertyCell(std::unique_ptr<LibCell> lib_cell)
  {
    _str2cell[lib_cell->get_cell_name()] = lib_cell.get();
    _cells.emplace_back(std::move(lib_cell));
  }

  LibCell* findCell(const char* cell_name)
  {
    auto p = _str2cell.find(cell_name);
    if (p != _str2cell.end()) {
      return p->second;
    }
    return nullptr;
  }

  void addLutTemplate(std::unique_ptr<LibLutTableTemplate> lut_template)
  {
    _str2template[lut_template->get_template_name()] = lut_template.get();
    _lut_templates.emplace_back(std::move(lut_template));
  }

  LibLutTableTemplate* getLutTemplate(const char* template_name)
  {
    auto p = _str2template.find(template_name);
    if (p != _str2template.end()) {
      return p->second;
    }
    return nullptr;
  }

  void addLibType(std::unique_ptr<LibType> lib_type)
  {
    _str2type[lib_type->get_type_name()] = lib_type.get();
    _types.emplace_back(std::move(lib_type));
  }

  LibType* getLibType(const char* lib_type_name)
  {
    auto p = _str2type.find(lib_type_name);
    if (p != _str2type.end()) {
      return p->second;
    }
    return nullptr;
  }

  void addWireLoad(std::unique_ptr<LibWireLoad> wire_load)
  {
    _str2wireLoad[wire_load->get_wire_load_name()] = wire_load.get();
    _wire_loads.emplace_back(std::move(wire_load));
  }

  LibWireLoad* getWireLoad(const char* wire_load_name)
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

  void set_time_unit(TimeUnit time_unit) { _time_unit = time_unit; }
  auto get_time_unit() { return _time_unit; }
  double convert_time_unit_to_ns(double src_value)
  {
    if (get_time_unit() == TimeUnit::kNS) {
      return src_value;
    } else if (get_time_unit() == TimeUnit::kPS) {
      return src_value * 1e-3;
    } else if (get_time_unit() == TimeUnit::kFS) {
      return src_value * 1e-6;
    }
    return 0.0;
  }

  std::vector<std::unique_ptr<LibCell>>& get_cells() { return _cells; }

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

  void printLibertyLibraryJson(const char* json_file_name);

 private:
  std::string _lib_name;
  std::vector<std::unique_ptr<LibCell>> _cells;  //!< The liberty cell, perserve the cell read order.
  StrMap<LibCell*> _str2cell;

  Vector<std::unique_ptr<LibLutTableTemplate>> _lut_templates;  //!< The timing table lut template, preserve the
                                                                //!< template order.

  StrMap<LibLutTableTemplate*> _str2template;

  Vector<std::unique_ptr<LibWireLoad>> _wire_loads;  //!< The wire load models.
  StrMap<LibWireLoad*> _str2wireLoad;

  Vector<std::unique_ptr<LibType>> _types;  //!< The lib type

  StrMap<LibType*> _str2type;

  CapacitiveUnit _cap_unit = CapacitiveUnit::kFF;
  ResistanceUnit _resistance_unit = ResistanceUnit::kkOHM;
  TimeUnit _time_unit = TimeUnit::kNS;

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

  FORBIDDEN_COPY(LibLibrary);
};

/**
 * @brief usage:
 * LibLibrary* lib;
 * LibCell* cell;
 * FOREACH_LIB_CELL(lib, cell)
 * {
 *    do_something_for_cell();
 * }
 *
 */
#define FOREACH_LIB_CELL(lib, cell)                                                     \
  for (std::vector<std::unique_ptr<LibCell>>::iterator iter = lib->get_cells().begin(); \
       iter != lib->get_cells().end() ? cell = (iter++->get()), true : false;)

/**
 * @brief The base class of liberty attribute value.
 * It would be string or float.
 *
 */
class LibAttrValue
{
 public:
  LibAttrValue() = default;
  virtual ~LibAttrValue() = default;

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
class LibFloatValue : public LibAttrValue
{
 public:
  explicit LibFloatValue(double val) : LibAttrValue(), _val(val) {}
  ~LibFloatValue() override = default;

  unsigned isFloat() override { return 1; }
  double getFloatValue() override { return _val; }

 private:
  double _val;
};

/**
 * @brief The liberty string value.
 *
 */
class LibStrValue : public LibAttrValue
{
 public:
  explicit LibStrValue(const char* val) : _val(val) {}
  ~LibStrValue() override = default;

  unsigned isString() override { return 1; }
  const char* getStringValue() override { return _val.c_str(); }

 private:
  std::string _val;
};

/**
 * @brief The library builder from the liberty to the sta analysis lib.
 *
 */
class LibBuilder
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
  explicit LibBuilder(const char* lib_name) : _lib(std::make_unique<LibLibrary>(lib_name)) {}
  ~LibBuilder() = default;

  LibLibrary* get_lib() { return _lib.get(); }
  std::unique_ptr<LibLibrary> takeLib() { return std::move(_lib); }

  void set_obj(LibObject* obj) { _obj = obj; }
  LibObject* get_obj() { return _obj; }

  void set_cell(LibCell* cell) { _obj = _cell = cell; }
  LibCell* get_cell() { return _cell; }

  void set_leakage_power(LibLeakagePower* leakage_power) { _obj = _leakage_power = leakage_power; }
  LibLeakagePower* get_leakage_power() { return _leakage_power; }

  void set_port(LibPort* port) { _obj = _port = port; }
  LibPort* get_port() { return _port; }

  void set_port_bus(LibPortBus* port) { _obj = _port = _port_bus = port; }
  LibPortBus* get_port_bus() { return _port_bus; }

  void set_arc(LibArc* arc) { _obj = _arc = arc; }
  LibArc* get_arc() { return _arc; }

  void set_power_arc(LibPowerArc* power_arc) { _obj = _power_arc = power_arc; }
  LibPowerArc* get_power_arc() { return _power_arc; }

  void set_table_model(LibTableModel* table_model) { _obj = _table_model = table_model; }
  LibTableModel* get_table_model() { return _table_model; }

  void set_table(LibTable* table) { _obj = _table = table; }
  LibTable* get_table() { return _table; }

  void set_current_table(LibCCSTable* table)
  {
    //@note pay attention to the obj and current table is not the same.
    // for we need get current table every time we parsed the vector.
    _current_table = table;
  }
  LibCCSTable* get_current_table() { return _current_table; }
  void set_own_port_type(LibertyOwnPortType own_port_type) { _own_port_type = own_port_type; }
  LibertyOwnPortType get_own_port_type() { return _own_port_type; }

  void set_own_pg_or_when_type(LibertyOwnPgOrWhenType own_pg_or_when_type) { _own_pg_or_when_type = own_pg_or_when_type; }
  LibertyOwnPgOrWhenType get_own_pg_or_when_type() { return _own_pg_or_when_type; }

 private:
  std::unique_ptr<LibLibrary> _lib;  //!< The current lib.

  LibObject* _obj = nullptr;                    //< The current library obj except the object below.
  LibCell* _cell = nullptr;                     //!< The parsed cell.
  LibLeakagePower* _leakage_power = nullptr;    //!< The parsed leakage power.
  LibPort* _port = nullptr;                     //!< The parsed port.
  LibPortBus* _port_bus = nullptr;              //!< The parsed port bus.
  LibArc* _arc = nullptr;                       //!< The parsed timing arc.
  LibPowerArc* _power_arc = nullptr;            //!< The parsed power arc.
  LibTableModel* _table_model = nullptr;        //!< The parsed table model.
  LibTable* _table = nullptr;                   //!< The parsed table.
  LibCCSTable* _current_table = nullptr;        //!< The parsed current table.
  LibertyOwnPortType _own_port_type;            //!< The flag of port own timing arc or power arc.
  LibertyOwnPgOrWhenType _own_pg_or_when_type;  //!< The flag of pg port/when own leakage power
                                                //!< or power arc.
  FORBIDDEN_COPY(LibBuilder);
};

/**
 * @brief This is the top interface class for liberty module.
 *
 */
class Lib
{
 public:
  Lib() = default;
  ~Lib() = default;

  RustLibertyReader loadLibertyWithRustParser(const char* file_name);

 private:
  FORBIDDEN_COPY(Lib);
};

}  // namespace ista
