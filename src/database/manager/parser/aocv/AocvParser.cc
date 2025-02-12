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
 * @file AocvParser.cc
 * @author longshy (longshy@pcl.ac.cn)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 */

#include "AocvParser.hh"

#include <sstream>

#include "Config.hh"

namespace ista {

AocvObjectSpec::AocvObjectSpec(const char* object_spec_name, AocvLibrary* own_aocv)
    : _object_spec_name(object_spec_name), _own_aocv(own_aocv)
{
}

AocvObjectSpec::AocvObjectSpec(AocvObjectSpec&& other) noexcept
    : _object_spec_name(other._object_spec_name),
      _trans_type(std::move(other._trans_type)),
      _delay_type(std::move(other._delay_type)),
      _analysis_mode(std::move(other._analysis_mode)),
      _path_type(std::move(other._path_type)),
      _depth2table(std::move(other._depth2table)),
      _default_table(std::move(other._default_table))
{
}

AocvObjectSpec& AocvObjectSpec::operator=(AocvObjectSpec&& rhs) noexcept
{
  if (this != &rhs) {
    _object_spec_name = rhs._object_spec_name;
    _trans_type = std::move(rhs._trans_type);
    _delay_type = std::move(rhs._delay_type);
    _analysis_mode = std::move(rhs._analysis_mode);
    _path_type = std::move(rhs._path_type);
    _depth2table = std::move(rhs._depth2table);
    _default_table = std::move(rhs._default_table);
  }

  return *this;
}

const std::map<std::string, TransType> AocvObjectSpec::_str2transtype
    = {{"rise", TransType::kRise}, {"fall", TransType::kFall}, {"rise fall", TransType::kRiseFall}};

const std::map<std::string, AnalysisMode> AocvObjectSpec::_str2analysisMode
    = {{"late", AnalysisMode::kMax}, {"early", AnalysisMode::kMin}, {"late early", AnalysisMode::kMaxMin}};

const std::map<std::string, AocvObjectSpec::DelayType> AocvObjectSpec::_str2delaytype = {{"cell", AocvObjectSpec::DelayType::kCell},
                                                                                         {"net", AocvObjectSpec::DelayType::kNet},
                                                                                         {"cell net", AocvObjectSpec::DelayType::kCellNet}};

const std::map<std::string, AocvObjectSpec::PathType> AocvObjectSpec::_str2pathtype
    = {{"clock", AocvObjectSpec::PathType::kClock},
       {"data", AocvObjectSpec::PathType::kData},
       {"clock data", AocvObjectSpec::PathType::kClockData}};

AocvObjectSpecSet::AocvObjectSpecSet(AocvObjectSpecSet&& other) noexcept : _object_specs(std::move(other._object_specs))
{
}

AocvObjectSpecSet& AocvObjectSpecSet::operator=(AocvObjectSpecSet&& rhs) noexcept
{
  if (this != &rhs) {
    _object_specs = std::move(rhs._object_specs);
  }
  return *this;
}

/**
 * @brief get the object spec according to the trans_type, analysis_mode and delay_type.
 *
 * @param rf_type
 * @param derate_type
 * @param delay_type
 * @return AocvObjectSpec*
 */
AocvObjectSpec* AocvObjectSpecSet::get_object_spec(TransType trans_type, AnalysisMode analysis_mode, AocvObjectSpec::DelayType delay_type)
{
  for (auto& object_spec : _object_specs) {
    if ((object_spec->get_trans_type() == trans_type) && (object_spec->get_analysis_mode() == analysis_mode)
        && (object_spec->get_delay_type() == delay_type)) {
      return object_spec.get();
    }
  }
  return nullptr;
}

/**
 * @brief find the all object specs with the same object_spec_name and the same
 * path_type.
 *
 * @param object_spec_name
 * @return std::optional<AocvObjectSpecSet*>
 */
std::optional<AocvObjectSpecSet*> AocvLibrary::findAocvObjectSpecSet(const char* object_spec_name, AocvObjectSpec::PathType path_type)
{
  auto object_specs = _obj_name_to_spec_set.values(object_spec_name);

  for (auto* object_spec : object_specs) {
    if (object_spec->front()->get_path_type() == path_type) {
      return object_spec;
    }
  }

  return std::nullopt;
}

/**
 * @brief find the all object specs with the same object_spec_name and the same
 * path_type(data).
 * @param object_spec_name
 * @return std::optional<AocvObjectSpecSet*>
 */
std::optional<AocvObjectSpecSet*> AocvLibrary::findDataAocvObjectSpecSet(const char* object_spec_name)
{
  auto object_spec_set = findAocvObjectSpecSet(object_spec_name, AocvObjectSpec::PathType::kData);

  return object_spec_set;
}

/**
 * @brief find the all object specs with the same object_spec_name and the same
 * path_type(clock).
 * @param object_spec_name
 * @return std::optional<AocvObjectSpecSet*>
 */
std::optional<AocvObjectSpecSet*> AocvLibrary::findClockAocvObjectSpecSet(const char* object_spec_name)
{
  auto object_spec_set = findAocvObjectSpecSet(object_spec_name, AocvObjectSpec::PathType::kClock);

  return object_spec_set;
}

/**
 * @brief add object spec to AocvLibrary.
 *
 * @param object_spec
 */
void AocvLibrary::addAocvObjectSpec(std::unique_ptr<AocvObjectSpec>&& object_spec)
{
  auto object_spec_set = findAocvObjectSpecSet(object_spec->get_object_spec_name(), object_spec->get_path_type());

  if (object_spec_set) {
    (*object_spec_set)->addAocvObjectSpec(std::move(object_spec));
  } else {
    auto* new_object_spec_set = new AocvObjectSpecSet();
    _object_spec_sets.emplace_back(new_object_spec_set);
    _obj_name_to_spec_set.insert(object_spec->get_object_spec_name(), new_object_spec_set);
    new_object_spec_set->addAocvObjectSpec(std::move(object_spec));
  }
}

/**
 * @brief  construct AocvObjectSpec by reading _stream(aocv file).
 *
 * @param current_line
 * @return std::unique_ptr<AocvObjectSpec>
 */
std::unique_ptr<AocvObjectSpec> AocvReader::readAocvObjectSpec(std::string current_line)
{
  auto object_spec = std::make_unique<AocvObjectSpec>();
  std::string::size_type pos_2 = current_line.find(':');
  std::string object_type = current_line.substr(pos_2 + 2);

  std::vector<int> depth_list;
  std::vector<float> table_list;
  std::unordered_map<int, float> depth2table;
  while (1) {
    // new_line:fit?
    std::string new_line;
    getline(_stream, new_line);
    if (new_line.find("rf_type") != std::string::npos) {
      std::string::size_type pos = new_line.find(':');
      std::string rf_type = new_line.substr(pos + 2);
      object_spec->set_trans_type(rf_type.c_str());
    } else if (new_line.find("delay_type") != std::string::npos) {
      std::string::size_type pos = new_line.find(':');
      std::string delay_type = new_line.substr(pos + 2);
      object_spec->set_delay_type(delay_type.c_str());
    } else if (new_line.find("derate_type") != std::string::npos) {
      std::string::size_type pos = new_line.find(':');
      std::string derate_type = new_line.substr(pos + 2);
      object_spec->set_analysis_mode(derate_type.c_str());
    } else if (new_line.find("path_type") != std::string::npos) {
      std::string::size_type pos = new_line.find(':');
      std::string path_type = new_line.substr(pos + 2);
      object_spec->set_path_type(path_type.c_str());
    } else if (new_line.find("object_spec") != std::string::npos) {
      std::string::size_type pos = new_line.find('/');
      std::string object_spec_name = new_line.substr(pos + 1);
      object_spec->set_object_spec_name(object_spec_name.c_str());
    } else if (new_line.find("depth") != std::string::npos) {
      if (new_line == "depth:") {
        object_spec->set_default_table(1);
        break;
      } else {
        std::string::size_type pos = new_line.find(':');
        std::string depth = new_line.substr(pos + 2);
        depth_list = strConvertNumList<int>(depth);
      }
    } else if (new_line.find("table") != std::string::npos) {
      std::string::size_type pos = new_line.find(':');
      std::string table = new_line.substr(pos + 2);
      table_list = strConvertNumList<float>(table);

      assert(depth_list.size() == table_list.size());
      for (size_t i = 0; i < depth_list.size(); i++) {
        depth2table[depth_list[i]] = table_list[i];
      }

      object_spec->set_depth2table(depth2table);
    }

    if (new_line.empty()) {
      break;
    }
  }

  return object_spec;
}

/**
 * @brief construct AocvLibrary by reading _stream(aocv file).
 *
 * @return std::unique_ptr<AocvLibrary>
 */
std::unique_ptr<AocvLibrary> AocvReader::readAocvLibrary()
{
  if (!_stream) {
    LOG_ERROR << "File " << _file_name << " NotReadable";
  }

  LOG_INFO << "start read aocv file " << _file_name;

  auto aocv_library = std::make_unique<AocvLibrary>(_file_name.c_str());

  // set version number.
  std::string new_line;
  getline(_stream, new_line);
  std::string version_number;
  if (new_line.find("version") != std::string::npos) {
    std::string::size_type pos_1 = new_line.find(':');
    version_number = new_line.substr(pos_1 + 2);
  }
  aocv_library->set_version_number(version_number.c_str());

  // set all aocv object specs.
  while (getline(_stream, new_line)) {
    if (new_line.find("object_type") != std::string::npos) {
      auto aocv_object_spec = readAocvObjectSpec(new_line);  // retVal

      aocv_library->addAocvObjectSpec(std::move(aocv_object_spec));
    }

    if (_stream.eof()) {
      LOG_INFO << "read aocv file EOF" << _file_name;
      break;
    }
  }

  return aocv_library;
}

/**
 * @brief convert string to num list.
 *
 * @tparam out_type
 * @param str
 * @return std::vector<out_type>
 */
template <typename out_type>
std::vector<out_type> AocvReader::strConvertNumList(std::string& str)
{
  std::stringstream stream(str);
  std::vector<out_type> num_list;
  out_type num;
  while (stream >> num) {
    num_list.push_back(num);
  }

  return num_list;
}

}  // namespace ista