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

#include <cstdint>

namespace ieda_feature {

class PLCommonSummary
{
 public:
  float place_density;
  float pin_density;
  int64_t PL_HPWL;
  int64_t PL_STWL;
  int64_t PL_GRWL;
  float congestion;
  float tns;
  float wns;
  float suggest_freq;

  // setter
  void setPlaceDensity(float density) { place_density = density; }
  void setPinDensity(float density) { pin_density = density; }
  void setPLHPWL(int64_t hpwl) { PL_HPWL = hpwl; }
  void setPLSTWL(int64_t stwl) { PL_STWL = stwl; }
  void setPLGRWL(int64_t grwl) { PL_GRWL = grwl; }
  void setCongestion(float value) { congestion = value; }
  void setTNS(float value) { tns = value; }
  void setWNS(float value) { wns = value; }
  void setSuggestFreq(float value) { suggest_freq = value; }

  // getter
  float getPlaceDensity() const { return place_density; }
  float getPinDensity() const { return pin_density; }
  int64_t getPLHPWL() const { return PL_HPWL; }
  int64_t getPLSTWL() const { return PL_STWL; }
  int64_t getPLGRWL() const { return PL_GRWL; }
  float getCongestion() const { return congestion; }
  float getTNS() const { return tns; }
  float getWNS() const { return wns; }
  float getSuggestFreq() const { return suggest_freq; }
};

class LGSummary
{
 public:
  PLCommonSummary pl_common_summary;
  int64_t lg_total_movement;
  int64_t lg_max_movement;
};

class PlaceSummary
{
 private:
  PLCommonSummary dplace;
  PLCommonSummary gplace;
  int32_t bin_number;
  int32_t bin_size_x;
  int32_t bin_size_y;
  int32_t core_width;
  int32_t core_height;
  int32_t fix_inst_cnt;
  int32_t instance_cnt;
  int32_t net_cnt;
  int32_t overflow_number;
  float overflow;
  int32_t total_pins;

  LGSummary lg_summary;

 public:
  // getter methods
  const PLCommonSummary& getDPlace() const { return dplace; }
  const PLCommonSummary& getGPlace() const { return gplace; }
  int32_t getBinNumber() const { return bin_number; }
  int32_t getBinSizeX() const { return bin_size_x; }
  int32_t getBinSizeY() const { return bin_size_y; }
  int32_t getCoreWidth() const { return core_width; }
  int32_t getCoreHeight() const { return core_height; }
  int32_t getFixInstCount() const { return fix_inst_cnt; }
  int32_t getInstanceCount() const { return instance_cnt; }
  int32_t getNetCount() const { return net_cnt; }
  int32_t getOverflowNumber() const { return overflow_number; }
  float getOverflow() const { return overflow; }
  int32_t getTotalPins() const { return total_pins; }
  const LGSummary& getLGSummary() const { return lg_summary; }

  // setter methods
  void setBinNumber(int32_t number) { bin_number = number; }
  void setBinSizeX(int32_t size) { bin_size_x = size; }
  void setBinSizeY(int32_t size) { bin_size_y = size; }
  void setCoreWidth(int32_t width) { core_width = width; }
  void setCoreHeight(int32_t height) { core_height = height; }
  void setFixInstCount(int32_t count) { fix_inst_cnt = count; }
  void setInstanceCount(int32_t count) { instance_cnt = count; }
  void setNetCount(int32_t count) { net_cnt = count; }
  void setOverflowNumber(int32_t number) { overflow_number = number; }
  void setOverflow(float value) { overflow = value; }
  void setTotalPins(int32_t count) { total_pins = count; }
  void setLGSummary(const LGSummary& summary) { lg_summary = summary; }
  // setter dp
  void setDPlace(const PLCommonSummary& summary) { dplace = summary; }
  void setDPlacedensity(float density) { dplace.setPlaceDensity(density); }
  void setDPlacePinDensity(float density) { dplace.setPinDensity(density); }
  void setDPlaceHPWL(int64_t hpwl) { dplace.setPLHPWL(hpwl); }
  void setDPlaceSTWL(int64_t stwl) { dplace.setPLSTWL(stwl); }
  void setDPlaceGRWL(int64_t grwl) { dplace.setPLGRWL(grwl); }
  void setDPlaceCongestion(float value) { dplace.setCongestion(value); }
  void setDPlaceTNS(float value) { dplace.setTNS(value); }
  void setDPlaceWNS(float value) { dplace.setWNS(value); }
  void setDPlaceSuggestFreq(float value) { dplace.setSuggestFreq(value); }

  // setter gp
  void setGPlace(const PLCommonSummary& summary) { gplace = summary; }
  void setGPlacedensity(float density) { gplace.setPlaceDensity(density); }
  void setGPlacePinDensity(float density) { gplace.setPinDensity(density); }
  void setGPlaceHPWL(int64_t hpwl) { gplace.setPLHPWL(hpwl); }
  void setGPlaceSTWL(int64_t stwl) { gplace.setPLSTWL(stwl); }
  void setGPlaceGRWL(int64_t grwl) { gplace.setPLGRWL(grwl); }
  void setGPlaceCongestion(float value) { gplace.setCongestion(value); }
  void setGPlaceTNS(float value) { gplace.setTNS(value); }
  void setGPlaceWNS(float value) { gplace.setWNS(value); }
  void setGPlaceSuggestFreq(float value) { gplace.setSuggestFreq(value); }

  // setter lg
  void setLGTotalMovement(int64_t movement) { lg_summary.lg_total_movement = movement; }
  void setLGMaxMovement(int64_t movement) { lg_summary.lg_max_movement = movement; }
  void setLGCommonSummary(const PLCommonSummary& summary) { lg_summary.pl_common_summary = summary; }
  void setLGPlacedensity(float density) { lg_summary.pl_common_summary.setPlaceDensity(density); }
  void setLGPlacePinDensity(float density) { lg_summary.pl_common_summary.setPinDensity(density); }
  void setLGPlaceHPWL(int64_t hpwl) { lg_summary.pl_common_summary.setPLHPWL(hpwl); }
  void setLGPlaceSTWL(int64_t stwl) { lg_summary.pl_common_summary.setPLSTWL(stwl); }
  void setLGPlaceGRWL(int64_t grwl) { lg_summary.pl_common_summary.setPLGRWL(grwl); }
  void setLGPlaceCongestion(float value) { lg_summary.pl_common_summary.setCongestion(value); }
  void setLGPlaceTNS(float value) { lg_summary.pl_common_summary.setTNS(value); }
  void setLGPlaceWNS(float value) { lg_summary.pl_common_summary.setWNS(value); }
  void setLGPlaceSuggestFreq(float value) { lg_summary.pl_common_summary.setSuggestFreq(value); }
};

}  // namespace ieda_feature