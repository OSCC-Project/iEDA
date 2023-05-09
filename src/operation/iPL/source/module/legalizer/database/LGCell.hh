/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-01 17:35:20
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-09 15:45:06
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGCell.hh
 * @Description: Cell Master data structure
 * 
 * 
 */
#ifndef IPL_LGCELL_H
#define IPL_LGCELL_H

#include <string>
#include <vector>

namespace ipl {

enum class LGCELL_TYPE
{
    kNone,
    kMacro,
    kSequence,
    kStdcell
};

class LGCell
{
public:
    LGCell() = delete;
    explicit LGCell(std::string name);
    LGCell(const LGCell&) = delete;
    LGCell(LGCell&&) = delete;
    ~LGCell();

    LGCell& operator=(const LGCell&) = delete;
    LGCell& operator=(LGCell&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    LGCELL_TYPE get_type() const { return _type;}
    int32_t get_width() const { return _width;}
    int32_t get_height() const { return _height;}

    // setter
    void set_type(LGCELL_TYPE type) { _type = type;}
    void set_width(int32_t width) { _width = width;}
    void set_height(int32_t height) { _height = height;}    

private:
    std::string _name;
    LGCELL_TYPE _type;
    int32_t _width;
    int32_t _height;
};
}
#endif