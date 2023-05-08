/*
 * @Author: S.J Chen
 * @Date: 2022-01-21 13:37:48
 * @LastEditTime: 2022-01-27 17:26:16
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/database/Orient.hh
 * Contact : https://github.com/sjchanson
 */

#ifndef IPL_ORIENT_H
#define IPL_ORIENT_H

#include "Rectangle.hh"
#include "module/logger/Log.hh"

namespace ipl {

enum class Orient : int8_t
{
  kNone,
  kN_R0,    /* Rotate object 0 degrees */
  kW_R90,   /* Rotate object 90 degrees */
  kS_R180,  /* Rotate object 180 degrees */
  kE_R270,  /* Rotate object 270 degrees */
  kFN_MY,   /* Mirror ablout the "Y" axis*/
  kFE_MY90, /* Mirror ablout the "Y" axis and rotate 90 degrees */
  kFS_MX,   /* Mirror ablout the "X" axis*/
  kFW_MX90  /* Mirror ablout the "X" axis and rotate 90 degrees */
};

}  // namespace ipl

#endif