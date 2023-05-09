/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-01 16:37:41
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-09 11:51:33
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGRow.hh
 * @Description: Row data structure
 * 
 * 
 */
#ifndef IPL_LGROW_H
#define IPL_LGROW_H

#include <string>
#include <vector>

#include "data/Rectangle.hh"
#include "data/Orient.hh"

namespace ipl {

class LGSite
{
public:
    LGSite() = delete;
    explicit LGSite(std::string name);
    LGSite(const LGSite&) = delete;
    LGSite(LGSite&&) = delete;
    ~LGSite();

    LGSite& operator=(const LGSite&) = delete;
    LGSite& operator=(LGSite&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    int32_t get_width() const { return _width;}
    int32_t get_height() const { return _height;}

    // setter
    void set_width(int32_t width) { _width = width;}
    void set_height(int32_t height) { _height = height;}

private:
    std::string _name;
    int32_t _width;
    int32_t _height;
};

class LGRow
{
public:
    LGRow() = delete;
    LGRow(std::string row_name, LGSite* site, int32_t site_num);
    LGRow(const LGRow&) = delete;
    LGRow(LGRow&&) = delete;
    ~LGRow();

    LGRow& operator=(const LGRow&) = delete;
    LGRow& operator=(LGRow&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    LGSite* get_site() const { return _site;}
    int32_t get_site_num() const { return _site_num;}
    Point<int32_t> get_coordinate() const { return _coordinate;}
    Orient get_row_orient() const { return _orient;}
    
    // setter
    void set_coordinate(int32_t lx, int32_t ly) {_coordinate = Point<int32_t>(lx,ly);}
    void set_orient(Orient orient) { _orient = orient;}

private:
    std::string _name;
    LGSite* _site;
    int32_t _site_num;
    Point<int32_t> _coordinate;
    Orient _orient;
};
}
#endif