/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-07 21:18:56
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-20 11:12:18
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGInterval.hh
 * @Description: LG interval data structure
 * 
 * 
 */
#ifndef IPL_LGINTERVAL_H
#define IPL_LGINTERVAL_H

#include <string>
#include "LGCluster.hh"

namespace ipl {
class LGRow;

class LGInterval
{
public:
    LGInterval() = delete;
    LGInterval(std::string name, int32_t min_x, int32_t max_x);
    LGInterval(const LGInterval&) = delete;
    LGInterval(LGInterval&&) = delete;
    ~LGInterval();

    LGInterval& operator=(const LGInterval&) = delete;
    LGInterval& operator=(LGInterval&&) = delete;

    // getter
    std::string get_name() const { return _name;}
    LGRow* get_belong_row() const { return _belong_row;}
    int32_t get_min_x()  { return _min_x;}
    int32_t get_max_x()  { return _max_x;}
    int32_t get_remain_length() const { return _remain_length;}
    LGCluster* get_cluster_root() const { return _cluster_root;}

    // setter
    void set_cluster_root(LGCluster* root){ _cluster_root = root;}
    void set_belong_row(LGRow* row) { _belong_row = row;}
    void set_min_x(int32_t min_x) { _min_x = min_x;}
    void set_max_x(int32_t max_x) { _max_x = max_x;}

    // function
    void updateRemainLength(int32_t occupied_length);
    void reset();

private:
    std::string _name; /* row_index + segment_index */
    LGCluster* _cluster_root;
    LGRow* _belong_row;

    int32_t _min_x;
    int32_t _max_x;
    int32_t _remain_length;
};
}
#endif