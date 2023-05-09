/*
 * @Author: Shijian Chen  chenshj@pcl.ac.cn
 * @Date: 2023-02-01 19:36:53
 * @LastEditors: Shijian Chen  chenshj@pcl.ac.cn
 * @LastEditTime: 2023-02-17 10:50:00
 * @FilePath: /irefactor/src/operation/iPL/source/module/legalizer_refactor/database/LGCluster.hh
 * @Description: Instance Clusters of LG
 * 
 * 
 */
#ifndef IPL_LGCLUSTER_H
#define IPL_LGCLUSTER_H

#include <string>
#include <vector>
#include "LGInstance.hh"

namespace ipl {

class LGInterval;

class LGCluster
{
public:
    LGCluster() = default;
    explicit LGCluster(std::string name);

    LGCluster(const LGCluster& other){
        _name = other._name;
        _inst_list = other._inst_list;
        _belong_segment = other._belong_segment;
        _min_x = other._min_x;
        _weight_e = other._weight_e;
        _weight_q = other._weight_q;
        _total_width = other._total_width;
        _front_cluster = other._front_cluster;
        _back_cluster = other._back_cluster;
    }
    LGCluster(LGCluster&& other){
        _name = std::move(other._name);
        _inst_list = std::move(other._inst_list);
        _belong_segment = std::move(other._belong_segment);
        _min_x = std::move(other._min_x);
        _weight_e = std::move(other._weight_e);
        _weight_q = std::move(other._weight_q);
        _total_width = std::move(other._total_width);
        _front_cluster = std::move(other._front_cluster);
        _back_cluster = std::move(other._back_cluster);
    }
    ~LGCluster();

    LGCluster& operator=(const LGCluster& other){
        _name = other._name;
        _inst_list = other._inst_list;
        _belong_segment = other._belong_segment;
        _min_x = other._min_x;
        _weight_e = other._weight_e;
        _weight_q = other._weight_q; 
        _total_width = other._total_width;
        _front_cluster = other._front_cluster;
        _back_cluster = other._back_cluster;
        return (*this);
    }
    LGCluster& operator=(LGCluster&& other){
        _name = std::move(other._name);
        _inst_list = std::move(other._inst_list);
        _belong_segment = std::move(other._belong_segment);
        _min_x = std::move(other._min_x);
        _weight_e = std::move(other._weight_e);
        _weight_q = std::move(other._weight_q);
        _total_width = std::move(other._total_width);
        _front_cluster = std::move(other._front_cluster);
        _back_cluster = std::move(other._back_cluster);
        return (*this);       
    }

    // getter
    std::string get_name() const { return _name;}
    std::vector<LGInstance*> get_inst_list() const { return _inst_list;}
    LGInterval* get_belong_interval() const { return _belong_segment;}
    int32_t get_min_x() const { return _min_x;}
    int32_t get_max_x();
    double get_weight_e() const { return _weight_e;}
    double get_weight_q() const { return _weight_q;}
    int32_t get_total_width() const { return _total_width;}
    LGCluster* get_front_cluster() const { return _front_cluster;}
    LGCluster* get_back_cluster() const { return _back_cluster;}

    // setter
    void set_name(std::string name) { _name = name;}
    void add_inst(LGInstance* inst) { _inst_list.push_back(inst);}
    void set_belong_interval(LGInterval* seg) { _belong_segment = seg;}
    void set_min_x(int32_t min_x) { _min_x = min_x;}
    void set_front_cluster(LGCluster* cluster) { _front_cluster = cluster;}
    void set_back_cluster(LGCluster* cluster) { _back_cluster = cluster;}

    // function
    void clearAbacusInfo();
    void insertInstance(LGInstance* inst);
    void appendCluster(LGCluster& cluster);
    void updateAbacusInfo(LGInstance* inst);

private:
    std::string _name;
    std::vector<LGInstance*> _inst_list;
    LGInterval* _belong_segment;

    int32_t _min_x;
    double _weight_e;
    double _weight_q;
    int32_t _total_width;

    LGCluster* _front_cluster;
    LGCluster* _back_cluster;

};
}
#endif