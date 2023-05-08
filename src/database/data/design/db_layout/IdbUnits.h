#pragma once
/**
 * @project		iDB
 * @file		IdbUnits.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        Describe Lef Units information, UNITS factors defined in lef ref as follow:
    [UNITS
        [TIME NANOSECONDS convertFactor ;]
        [CAPACITANCE PICOFARADS convertFactor ;]
        [RESISTANCE OHMS convertFactor ;]
        [POWER MILLIWATTS convertFactor ;]
        [CURRENT MILLIAMPS convertFactor ;]
        [VOLTAGE VOLTS convertFactor ;]
        [DATABASE MICRONS LEFconvertFactor ;]
        [FREQUENCY MEGAHERTZ convertFactor ;]
    END UNITS]
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>

namespace idb {

using std::vector;

class IdbUnits
{
 public:
  IdbUnits();
  ~IdbUnits() = default;

  // getter
  const int32_t get_nanoseconds() const { return _nanoseconds; }
  const int32_t get_picofarads() const { return _picofarads; }
  const int32_t get_ohms() const { return _ohms; }
  const int32_t get_milliwatts() const { return _milliwatts; }
  const int32_t get_milliamps() const { return _milliamps; }
  const int32_t get_volts() const { return _volts; }
  const int32_t get_micron_dbu() const { return _micron_dbu; }
  const int32_t get_megahertz() const { return _megahertz; }

  // setter
  void set_nanoseconds(int32_t nanoseconds) { _nanoseconds = nanoseconds; }
  void set_picofarads(int32_t picofarads) { _picofarads = picofarads; }
  void set_ohms(int32_t ohms) { _ohms = ohms; }
  void set_milliwatts(int32_t milliwatts) { _milliwatts = milliwatts; }
  void set_milliamps(int32_t milliamps) { _milliamps = milliamps; }
  void set_volts(int32_t volts) { _volts = volts; }
  void set_microns_dbu(int32_t micron_dbu) { _micron_dbu = micron_dbu; }
  void set_megahertz(int32_t megahertz) { _megahertz = megahertz; }

  // operator

  // verify data
  void print();

 private:
  int32_t _nanoseconds;
  int32_t _picofarads;
  int32_t _ohms;
  int32_t _milliwatts;
  int32_t _milliamps;
  int32_t _volts;
  int32_t _micron_dbu;
  int32_t _megahertz;
};

}  // namespace idb
