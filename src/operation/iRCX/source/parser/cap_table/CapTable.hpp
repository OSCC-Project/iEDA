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
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
namespace ircx {
namespace parser {

/**
 * @brief One lookup row from a captab configuration.
 *
 * Each row is encoded as:
 *   spacing coupling_cap ground_cap
 */
struct CapTableEntry {
  double distance{0.0};      // same-layer spacing (um)
  double coupling_cap{0.0};  // coupling capacitance (fF/um)
  double ground_cap{0.0};    // ground capacitance (fF/um)
};

/**
 * @brief All lookup rows for one captab environment.
 *
 * Key examples:
 * - A-series (no upper layer): "A M2 OVER M1"
 * - A-series (to substrate):   "A M1 OVER SUBSTRATE"
 * - B-series (upper+lower):    "B M2 OVER M1 UNDER M3"
 */
struct CapTableConfig {
  std::string key;
  std::string type;
  std::string layer_name;
  std::string over_layer;
  std::string under_layer;
  std::vector<CapTableEntry> data;
};

/**
 * @brief Capacitance lookup result.
 */
struct CapacitanceResult {
  double ground_cap{0.0};
  double coupling_cap{0.0};
};

/**
 * @brief Unified captab parser, interpolator, and query API.
 *
 * Supports only the new captab format:
 * ```
 * A M1 OVER SUBSTRATE
 * 0.045 0.098217 0.014883
 * 0.090 0.056322 0.016892
 * ...
 *
 * B M1 OVER SUBSTRATE UNDER M2
 * 0.045 0.025255 0.092115
 * 0.090 0.032732 0.048268
 * ...
 * ```
 */
class CapTable {
 public:
  CapTable() = default;
  ~CapTable() = default;

  // Parsing

  /**
   * @brief Load captab content from a file.
   */
  bool loadFromFile(const std::string& filePath);

  /**
   * @brief Load captab content from a string buffer.
   */
  bool loadFromString(const std::string& content);

  /**
   * @brief Return all stored captab keys.
   */
  std::vector<std::string> get_all_keys() const;

  /**
   * @brief Number of stored captab configurations.
   */
  size_t size() const { return configs_.size(); }

  // Queries

  /**
   * @brief Query an A-series table (two-layer context).
   *
   * The lookup distance is the non-negative same-layer spacing in microns.
   * For isolated/no-neighbor cases, use queryTwoLayerIsolatedCap().
   *
   * Negative distances are treated as a legacy compatibility path and map
   * to the isolated result.
   */
  CapacitanceResult queryTwoLayerCap(
      const std::string& layer_name,
      const std::string& belowLayer,
      double neighborDistance) const;

  /**
   * @brief Query a B-series table (three-layer context).
   *
   * The lookup distance is the non-negative same-layer spacing in microns.
   * For isolated/no-neighbor cases, use queryThreeLayerIsolatedCap().
   *
   * Negative distances are treated as a legacy compatibility path and map
   * to the isolated result.
   */
  CapacitanceResult queryThreeLayerCap(
      const std::string& layer_name,
      const std::string& belowLayer,
      const std::string& aboveLayer,
      double neighborDistance) const;

  /**
   * @brief Query the isolated/fringe result for an A-series table.
   */
  CapacitanceResult queryTwoLayerIsolatedCap(
      const std::string& layer_name,
      const std::string& belowLayer) const;

  /**
   * @brief Query the isolated/fringe result for a B-series table.
   */
  CapacitanceResult queryThreeLayerIsolatedCap(
      const std::string& layer_name,
      const std::string& belowLayer,
      const std::string& aboveLayer) const;

  /**
   * @brief Query the farthest lookup row and keep both ground/coupling terms.
   *
   * Used by open-ended calculation formulas that need the asymptotic `cg + cc`
   * contribution instead of the isolated ground-only fallback.
   */
  CapacitanceResult queryTwoLayerFarthestCap(
      const std::string& layer_name,
      const std::string& belowLayer) const;

  /**
   * @brief Query the farthest lookup row and keep both ground/coupling terms.
   */
  CapacitanceResult queryThreeLayerFarthestCap(
      const std::string& layer_name,
      const std::string& belowLayer,
      const std::string& aboveLayer) const;

 private:
  // Parsing helpers

  bool parseConfig(const std::string& headerLine,
                   const std::vector<std::string>& dataLines);
  static bool parseConfig(const std::string& headerLine,
                          const std::vector<std::string>& dataLines,
                          std::map<std::string, CapTableConfig>& configs);

  static std::string makeKey(const std::string& layer_name,
                             const std::string& overLayer,
                             const std::string& underLayer);

  const CapTableConfig* get_config(
      const std::string& layer_name,
      const std::string& overLayer,
      const std::string& underLayer = "") const;

  // Query helpers

  CapacitanceResult interpolate(
      const std::string& layer_name,
      const std::string& overLayer,
      const std::string& underLayer,
      double neighborDistance) const;

  CapacitanceResult farthestResult(const CapTableConfig& config) const;
  CapacitanceResult isolatedResult(const CapTableConfig& config) const;

  double linearInterpolate(double x, double x1, double y1, double x2, double y2) const;

  std::pair<int, int> findBracketingIndices(
      const std::vector<CapTableEntry>& data,
      double distance) const;

 private:
  std::map<std::string, CapTableConfig> configs_;
};

}  // namespace parser
}  // namespace ircx
