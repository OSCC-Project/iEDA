#pragma once

#include <assert.h>

#include <string>

namespace idb {

enum class GdsFormatType
{
  kGDSII_Archive = 0b01,
  kGDSII_Filtered = 0b10,
  kEDSM_Archive = 0b11,
  kEDSHI_Filtered = 0b100,
};

class GdsFormat
{
 public:
  GdsFormat() : type(GdsFormatType::kGDSII_Archive), mask() {}

  bool is_archive() const;
  bool is_filtered() const;

  GdsFormatType type;
  std::string mask;
};

//////////// inline ///////

inline bool GdsFormat::is_archive() const
{
  return (int) type & 0b1;
}

inline bool GdsFormat::is_filtered() const
{
  return !is_archive();
}

}  // namespace idb