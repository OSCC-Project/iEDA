/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-03-01 17:48:56
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-03-03 21:00:03
 * @FilePath: /irefactor/src/operation/iPL/source/module/detail_refactor/database/DPCell.hh
 * @Description: Cell Master of Instance
 * 
 * 
 */
#ifndef IPL_DPCELL_H
#define IPL_DPCELL_H

#include <string>

namespace ipl {

enum class DPCELL_TYPE
{
    kNone,
    kMacro,
    kSequence,
    kStdcell
};

class DPCell
{
public:
    DPCell() = delete;
    explicit DPCell(std::string name);
    DPCell(const DPCell&) = delete;
    DPCell(DPCell&&) = delete;
    ~DPCell();

    DPCell& operator=(const DPCell&) = delete;
    DPCell& operator=(DPCell&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    DPCELL_TYPE get_type() const { return _type;}
    int32_t get_width() const { return _width;}
    int32_t get_height() const { return _height;}

    // setter
    void set_type(DPCELL_TYPE type) { _type = type;}
    void set_width(int32_t width) { _width = width;}
    void set_height(int32_t height) { _height = height;} 

private:
    std::string _name;
    int32_t _width;
    int32_t _height;
    DPCELL_TYPE _type;
};
}
#endif