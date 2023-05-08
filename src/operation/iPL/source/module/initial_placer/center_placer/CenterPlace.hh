/*
 * @Author: S.J Chen
 * @Date: 2022-04-01 11:49:58
 * @LastEditTime: 2022-04-01 12:04:45
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/pre_placer/center_place/CenterPlace.hh
 * Contact : https://github.com/sjchanson
 */
#ifndef IPL_OPERATOR_PP_CENTER_PLACE_H
#define IPL_OPERATOR_PP_CENTER_PLACE_H

#include "Log.hh"
#include "PlacerDB.hh"

namespace ipl {

class CenterPlace
{
 public:
  CenterPlace() = delete;
  explicit CenterPlace(PlacerDB* placer_db);
  CenterPlace(const CenterPlace&) = delete;
  CenterPlace(CenterPlace&&) = delete;
  ~CenterPlace() = default;

  CenterPlace& operator=(const CenterPlace&) = delete;
  CenterPlace& operator=(CenterPlace&&) = delete;

  void runCenterPlace();

 private:
  PlacerDB* _placer_db;
};
inline CenterPlace::CenterPlace(PlacerDB* placer_db) : _placer_db(placer_db)
{
}

}  // namespace ipl

#endif