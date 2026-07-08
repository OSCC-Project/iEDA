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
 * @file GdsStream.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-28
 * @brief Minimal binary GDSII stream writer implementation for CTS clock-tree visualization files.
 */

#include "report/visualization/gds/writer/GdsStream.hh"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

#include "Log.hh"
#include "Point.hh"

namespace icts::visualization {
namespace {

enum class GdsRecord : uint8_t
{
  kHeader = 0x00,
  kBgnLib = 0x01,
  kLibName = 0x02,
  kUnits = 0x03,
  kEndLib = 0x04,
  kBgnStr = 0x05,
  kStrName = 0x06,
  kEndStr = 0x07,
  kBoundary = 0x08,
  kPath = 0x09,
  kText = 0x0C,
  kLayer = 0x0D,
  kDataType = 0x0E,
  kWidth = 0x0F,
  kXy = 0x10,
  kEndEl = 0x11,
  kTextType = 0x16,
  kString = 0x19
};

enum class GdsDataType : uint8_t
{
  kNoData = 0x00,
  kInt2 = 0x02,
  kInt4 = 0x03,
  kReal8 = 0x05,
  kAscii = 0x06
};

auto ensureParentDirectory(const std::filesystem::path& path) -> bool
{
  const auto parent_path = path.parent_path();
  if (parent_path.empty()) {
    return true;
  }

  std::error_code error;
  std::filesystem::create_directories(parent_path, error);
  if (error) {
    LOG_WARNING << "GdsStream: failed to create directory " << parent_path.string() << ": " << error.message();
    return false;
  }
  return true;
}

auto appendInt16(std::vector<uint8_t>& bytes, int16_t value) -> void
{
  const auto unsigned_value = static_cast<uint16_t>(value);
  bytes.push_back(static_cast<uint8_t>((unsigned_value >> 8U) & 0xFFU));
  bytes.push_back(static_cast<uint8_t>(unsigned_value & 0xFFU));
}

auto appendInt32(std::vector<uint8_t>& bytes, int32_t value) -> void
{
  const auto unsigned_value = static_cast<uint32_t>(value);
  bytes.push_back(static_cast<uint8_t>((unsigned_value >> 24U) & 0xFFU));
  bytes.push_back(static_cast<uint8_t>((unsigned_value >> 16U) & 0xFFU));
  bytes.push_back(static_cast<uint8_t>((unsigned_value >> 8U) & 0xFFU));
  bytes.push_back(static_cast<uint8_t>(unsigned_value & 0xFFU));
}

auto makeGdsReal8(double value) -> std::array<uint8_t, 8>
{
  std::array<uint8_t, 8> bytes{};
  if (value == 0.0 || !std::isfinite(value)) {
    return bytes;
  }

  bool negative = value < 0.0;
  double normalized = std::fabs(value);
  int exponent = 64;
  while (normalized >= 1.0) {
    normalized /= 16.0;
    ++exponent;
  }
  while (normalized < 0.0625) {
    normalized *= 16.0;
    --exponent;
  }

  if (exponent <= 0 || exponent >= 128) {
    LOG_WARNING << "GdsStream: real8 value is outside GDSII exponent range.";
    return bytes;
  }

  auto mantissa = static_cast<uint64_t>(std::ldexp(normalized, 56));
  if (mantissa >= (uint64_t{1} << 56U)) {
    mantissa = (uint64_t{1} << 56U) - 1U;
  }

  bytes[0] = static_cast<uint8_t>((negative ? 0x80U : 0x00U) | static_cast<uint8_t>(exponent));
  for (std::size_t byte_index = 7U; byte_index > 0U; --byte_index) {
    bytes[byte_index] = static_cast<uint8_t>(mantissa & 0xFFU);
    mantissa >>= 8U;
  }
  return bytes;
}

template <typename Bytes>
auto writeBytes(std::ofstream& stream, const Bytes& bytes) -> void
{
  for (const auto byte : bytes) {
    stream.put(static_cast<char>(byte));
  }
}

auto writeRecord(std::ofstream& stream, GdsRecord record, GdsDataType data_type, const std::vector<uint8_t>& payload) -> bool
{
  const auto length = payload.size() + 4U;
  if (length > static_cast<std::size_t>(std::numeric_limits<uint16_t>::max())) {
    LOG_WARNING << "GdsStream: record payload is too large for one GDSII record.";
    return false;
  }

  std::array<uint8_t, 4> header{
      static_cast<uint8_t>((length >> 8U) & 0xFFU),
      static_cast<uint8_t>(length & 0xFFU),
      static_cast<uint8_t>(record),
      static_cast<uint8_t>(data_type),
  };
  writeBytes(stream, header);
  if (!payload.empty()) {
    writeBytes(stream, payload);
  }
  return static_cast<bool>(stream);
}

auto writeNoDataRecord(std::ofstream& stream, GdsRecord record) -> bool
{
  return writeRecord(stream, record, GdsDataType::kNoData, {});
}

auto writeInt16Record(std::ofstream& stream, GdsRecord record, const std::vector<int16_t>& values) -> bool
{
  std::vector<uint8_t> payload;
  payload.reserve(values.size() * 2U);
  for (const auto value : values) {
    appendInt16(payload, value);
  }
  return writeRecord(stream, record, GdsDataType::kInt2, payload);
}

auto writeInt32Record(std::ofstream& stream, GdsRecord record, const std::vector<int32_t>& values) -> bool
{
  std::vector<uint8_t> payload;
  payload.reserve(values.size() * 4U);
  for (const auto value : values) {
    appendInt32(payload, value);
  }
  return writeRecord(stream, record, GdsDataType::kInt4, payload);
}

auto writeReal8Record(std::ofstream& stream, GdsRecord record, const std::vector<double>& values) -> bool
{
  std::vector<uint8_t> payload;
  payload.reserve(values.size() * 8U);
  for (const auto value : values) {
    const auto real_bytes = makeGdsReal8(value);
    payload.insert(payload.end(), real_bytes.begin(), real_bytes.end());
  }
  return writeRecord(stream, record, GdsDataType::kReal8, payload);
}

auto writeStringRecord(std::ofstream& stream, GdsRecord record, const std::string& value) -> bool
{
  std::vector<uint8_t> payload(value.begin(), value.end());
  if ((payload.size() % 2U) != 0U) {
    payload.push_back(0U);
  }
  return writeRecord(stream, record, GdsDataType::kAscii, payload);
}

auto makeDatePayload() -> std::vector<int16_t>
{
  return {
      2026, 4, 28, 0, 0, 0, 2026, 4, 28, 0, 0, 0,
  };
}

auto writeLayerDatatype(std::ofstream& stream, const GdsLayerKey& key, bool text) -> bool
{
  return writeInt16Record(stream, GdsRecord::kLayer, {key.layer})
         && writeInt16Record(stream, text ? GdsRecord::kTextType : GdsRecord::kDataType, {key.datatype});
}

auto flattenPoints(const std::vector<Point<int>>& points) -> std::vector<int32_t>
{
  std::vector<int32_t> xy;
  xy.reserve(points.size() * 2U);
  for (const auto& point : points) {
    xy.push_back(point.get_x());
    xy.push_back(point.get_y());
  }
  return xy;
}

auto writePath(std::ofstream& stream, const GdsPath& path) -> bool
{
  if (path.points.size() < 2U) {
    return true;
  }
  return writeNoDataRecord(stream, GdsRecord::kPath) && writeLayerDatatype(stream, path.key, false)
         && writeInt32Record(stream, GdsRecord::kWidth, {std::max(path.width_dbu, int32_t{1})})
         && writeInt32Record(stream, GdsRecord::kXy, flattenPoints(path.points)) && writeNoDataRecord(stream, GdsRecord::kEndEl);
}

auto writeBoundary(std::ofstream& stream, const GdsBoundary& boundary) -> bool
{
  if (boundary.points.size() < 3U) {
    return true;
  }

  auto points = boundary.points;
  if (points.front().get_x() != points.back().get_x() || points.front().get_y() != points.back().get_y()) {
    points.push_back(points.front());
  }
  return writeNoDataRecord(stream, GdsRecord::kBoundary) && writeLayerDatatype(stream, boundary.key, false)
         && writeInt32Record(stream, GdsRecord::kXy, flattenPoints(points)) && writeNoDataRecord(stream, GdsRecord::kEndEl);
}

auto writeText(std::ofstream& stream, const GdsText& text) -> bool
{
  if (text.text.empty()) {
    return true;
  }
  return writeNoDataRecord(stream, GdsRecord::kText) && writeLayerDatatype(stream, text.key, true)
         && writeInt32Record(stream, GdsRecord::kXy, {text.origin.get_x(), text.origin.get_y()})
         && writeStringRecord(stream, GdsRecord::kString, text.text) && writeNoDataRecord(stream, GdsRecord::kEndEl);
}

auto escapeXml(const std::string& text) -> std::string
{
  std::string escaped;
  escaped.reserve(text.size());
  for (const char ch : text) {
    switch (ch) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      case '"':
        escaped += "&quot;";
        break;
      case '\'':
        escaped += "&apos;";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

}  // namespace

auto GdsStream::writeBinary(const std::filesystem::path& path, const GdsLibrary& library) -> bool
{
  if (!ensureParentDirectory(path)) {
    return false;
  }

  std::ofstream stream(path, std::ios::binary);
  if (!stream.is_open()) {
    LOG_WARNING << "GdsStream: failed to open " << path.string() << " for binary GDSII output.";
    return false;
  }

  const double dbu_per_um = static_cast<double>(std::max(library.dbu_per_um, int32_t{1}));
  const double database_unit_in_user_units = 1.0 / dbu_per_um;
  const double database_unit_in_meters = 1.0e-6 / dbu_per_um;

  bool success = writeInt16Record(stream, GdsRecord::kHeader, {600}) && writeInt16Record(stream, GdsRecord::kBgnLib, makeDatePayload())
                 && writeStringRecord(stream, GdsRecord::kLibName, library.library_name)
                 && writeReal8Record(stream, GdsRecord::kUnits, {database_unit_in_user_units, database_unit_in_meters})
                 && writeInt16Record(stream, GdsRecord::kBgnStr, makeDatePayload())
                 && writeStringRecord(stream, GdsRecord::kStrName, library.structure_name);

  for (const auto& boundary : library.boundaries) {
    success = success && writeBoundary(stream, boundary);
  }
  for (const auto& path_element : library.paths) {
    success = success && writePath(stream, path_element);
  }
  for (const auto& text : library.texts) {
    success = success && writeText(stream, text);
  }

  success = success && writeNoDataRecord(stream, GdsRecord::kEndStr) && writeNoDataRecord(stream, GdsRecord::kEndLib);
  stream.close();
  return success && static_cast<bool>(stream);
}

auto GdsStream::writeLayerProperties(const std::filesystem::path& path, const std::vector<GdsLayerProperty>& layers) -> bool
{
  if (!ensureParentDirectory(path)) {
    return false;
  }

  std::ofstream stream(path);
  if (!stream.is_open()) {
    LOG_WARNING << "GdsStream: failed to open " << path.string() << " for layer properties.";
    return false;
  }

  stream << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
  stream << "<layer-properties>\n";
  for (const auto& layer : layers) {
    stream << "  <properties>\n";
    stream << "    <frame-color>" << escapeXml(layer.color) << "</frame-color>\n";
    stream << "    <fill-color>" << escapeXml(layer.color) << "</fill-color>\n";
    stream << "    <frame-brightness>0</frame-brightness>\n";
    stream << "    <fill-brightness>0</fill-brightness>\n";
    stream << "    <dither-pattern>I5</dither-pattern>\n";
    stream << "    <visible>true</visible>\n";
    stream << "    <transparent>false</transparent>\n";
    stream << "    <width>1</width>\n";
    stream << "    <marked>false</marked>\n";
    stream << "    <xfill>false</xfill>\n";
    stream << "    <animation>0</animation>\n";
    stream << "    <name>" << escapeXml(layer.name) << "</name>\n";
    stream << "    <source>" << layer.key.layer << "/" << layer.key.datatype << "@1</source>\n";
    stream << "  </properties>\n";
  }
  stream << "</layer-properties>\n";
  stream.close();
  return static_cast<bool>(stream);
}

}  // namespace icts::visualization
