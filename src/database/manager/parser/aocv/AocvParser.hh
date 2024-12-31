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
 * @file AocvParser.hh
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 */
#pragma once

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "BTreeMap.hh"
#include "include/Type.hh"

namespace ista {

class AocvLibrary;

/**
 * @brief The object spec in the aocv.
 *
 */
class AocvObjectSpec
{
 public:
  enum class DelayType : int
  {
    kCell = 1,
    kNet = 2,
    kCellNet = 3
  };

  enum class PathType : int
  {
    kClock = 1,
    kData = 2,
    kClockData = 3
  };

  AocvObjectSpec(const char* object_spec_name, AocvLibrary* own_aocv);
  AocvObjectSpec() = default;
  ~AocvObjectSpec() = default;

  AocvObjectSpec(AocvObjectSpec&& other) noexcept;
  AocvObjectSpec& operator=(AocvObjectSpec&& rhs) noexcept;

  // setter
  void set_object_spec_name(const char* object_spec_name) { _object_spec_name = object_spec_name; }
  void set_object_type(const char* object_type) { _object_type = object_type; }
  void set_trans_type(const char* trans_type) { _trans_type = _str2transtype.at(trans_type); }
  void set_delay_type(const char* delay_type) { _delay_type = _str2delaytype.at(delay_type); }
  void set_analysis_mode(const char* analysis_mode) { _analysis_mode = _str2analysisMode.at(analysis_mode); }
  void set_path_type(const char* path_type) { _path_type = _str2pathtype.at(path_type); }
  void set_depth2table(std::unordered_map<int, float>& depth2table) { _depth2table = depth2table; }
  void set_default_table(int default_table) { _default_table = default_table; }
  void set_own_aocv(AocvLibrary* own_aocv) { _own_aocv = own_aocv; }

  // getter
  const char* get_object_spec_name() { return _object_spec_name.c_str(); }
  const char* get_object_type() { return _object_type.c_str(); }
  TransType get_trans_type() { return _trans_type; }
  DelayType get_delay_type() { return _delay_type; }
  AnalysisMode get_analysis_mode() { return _analysis_mode; }
  PathType get_path_type() { return _path_type; }
  auto& get_depth2table() { return _depth2table; }
  std::optional<int> get_default_table() { return _default_table; }
  AocvLibrary* get_own_aocv() { return _own_aocv; }

 protected:
  static const std::map<std::string, TransType> _str2transtype;
  static const std::map<std::string, AnalysisMode> _str2analysisMode;
  static const std::map<std::string, DelayType> _str2delaytype;
  static const std::map<std::string, PathType> _str2pathtype;

 private:
  std::string _object_spec_name;
  std::string _object_type;
  TransType _trans_type;
  DelayType _delay_type;
  AnalysisMode _analysis_mode;
  PathType _path_type;
  std::optional<std::unordered_map<int, float>> _depth2table;
  std::optional<int> _default_table;
  AocvLibrary* _own_aocv;

  FORBIDDEN_COPY(AocvObjectSpec);
};

/**
 * @brief The aocv object spec have the same name, except the condition is
 * different.
 *
 */
class AocvObjectSpecSet
{
 public:
  AocvObjectSpecSet() = default;
  ~AocvObjectSpecSet() = default;
  AocvObjectSpecSet(AocvObjectSpecSet&& other) noexcept;
  AocvObjectSpecSet& operator=(AocvObjectSpecSet&& rhs) noexcept;

  void addAocvObjectSpec(std::unique_ptr<AocvObjectSpec>&& object_spec) { _object_specs.emplace_back(std::move(object_spec)); }
  AocvObjectSpec* front() { return !_object_specs.empty() ? _object_specs.front().get() : nullptr; }
  AocvObjectSpec* get_object_spec(TransType trans_type, AnalysisMode analysis_mode, AocvObjectSpec::DelayType delay_type);
  auto& get_object_specs() { return _object_specs; }

 private:
  std::vector<std::unique_ptr<AocvObjectSpec>> _object_specs;

  FORBIDDEN_COPY(AocvObjectSpecSet);
};

/**
 * @brief The aocv libraty class.
 *
 */
class AocvLibrary
{
 public:
  explicit AocvLibrary(const char* aocv_name) : _aocv_name(aocv_name) {}
  AocvLibrary() = default;
  ~AocvLibrary() = default;

  void set_version_number(const char* version_number) { _version_number = version_number; }
  const char* get_version_number() const { return _version_number.c_str(); }
  auto& get_object_spec_sets() { return _object_spec_sets; }

  std::optional<AocvObjectSpecSet*> findAocvObjectSpecSet(const char* object_spec_name, AocvObjectSpec::PathType path_type);
  std::optional<AocvObjectSpecSet*> findDataAocvObjectSpecSet(const char* object_spec_name);
  std::optional<AocvObjectSpecSet*> findClockAocvObjectSpecSet(const char* object_spec_name);
  void addAocvObjectSpec(std::unique_ptr<AocvObjectSpec>&& object_spec);

 private:
  std::string _version_number;                                                 //!< The version of the aocv file.
  std::string _aocv_name;                                                      //!< The name of the aocv file.
  std::vector<std::unique_ptr<AocvObjectSpecSet>> _object_spec_sets;           //!< The all object specs of the aocv file.
  ieda::Multimap<std::string_view, AocvObjectSpecSet*> _obj_name_to_spec_set;  //!< The obj name map to the object spec set.

  FORBIDDEN_COPY(AocvLibrary);
};

/**
 * @brief The aocv reader is used to read the related keyword.
 *
 */
class AocvReader
{
 public:
  AocvReader(const char* file_name) : _file_name(file_name) { _stream.open(file_name); }
  ~AocvReader() { _stream.close(); };

  std::unique_ptr<AocvObjectSpec> readAocvObjectSpec(std::string current_line);
  std::unique_ptr<AocvLibrary> readAocvLibrary();

  template <typename out_type>
  std::vector<out_type> strConvertNumList(std::string& str);

 private:
  std::string _file_name;  //!< The aocv file name.
  int _line_no = 0;        //!< The aocv file line no.
  std::ifstream _stream;
};

}  // namespace ista