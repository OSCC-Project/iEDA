/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-03 18:58:21
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-07 10:38:41
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPDatabase.hh
 * @Description: Database of detail placement
 * 
 * 
 */
#ifndef IPL_DPDATABASE_H
#define IPL_DPDATABASE_H

#include <string>

#include "PlacerDB.hh"
#include "DPDesign.hh"
#include "DPLayout.hh"


namespace ipl {
class DPDatabase
{
public:
    DPDatabase();
    
    DPDatabase(const DPDatabase&) = delete;
    DPDatabase(DPDatabase&&) = delete;
    ~DPDatabase();

    DPDatabase& operator=(const DPDatabase&) = delete;
    DPDatabase& operator=(DPDatabase&&) = delete;

    // getter
    DPDesign* get_design() const { return _design;}
    DPLayout* get_layout() const { return _layout;}

    int64_t get_outside_wl() const { return _outside_wl;}

private:
    PlacerDB* _placer_db;
    int32_t _shift_x;
    int32_t _shift_y;
    int64_t _outside_wl;

    DPDesign* _design;
    DPLayout* _layout;

    friend class DetailPlacer;
};
}
#endif