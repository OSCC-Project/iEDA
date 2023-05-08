/**
 * @project		iDB
 * @file		IdbUnits.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe lef Units information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbUnits.h"

namespace idb {

IdbUnits::IdbUnits()
{
  _nanoseconds = -1;
  _picofarads = -1;
  _ohms = -1;
  _milliwatts = -1;
  _milliamps = -1;
  _volts = -1;
  _micron_dbu = -1;
  _megahertz = -1;
}

void IdbUnits::print()
{
  std::cout << "nanoseconds = " << _nanoseconds << " picofarads = " << _picofarads << " ohms = "
            << " milliwatts = " << _milliwatts << " milliamps = " << _milliamps << " volts = " << _volts << " micron_dbu = " << _micron_dbu
            << " megahertz = " << _megahertz << std::endl;
}

}  // namespace idb
