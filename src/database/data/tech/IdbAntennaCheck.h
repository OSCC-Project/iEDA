#ifndef _IDB_ANTENNA_CHECK_H
#define _IDB_ANTENNA_CHECK_H

#include <map>

namespace idb {
  class IdbAntennaCheck {
   public:
    IdbAntennaCheck()
        : _antenna_cum_routing_plus_cut(false),
          _has_antenna_diff_area_ratio_pwl(false),
          _has_antenna_cum_diff_area_ratio_pwl(false),
          _antenna_area_ratio(-1),
          _antenna_cum_area_ratio(-1) { }
    ~IdbAntennaCheck() { }
    // getter
    bool get_antenna_cum_routing_plus_cut() { return _antenna_cum_routing_plus_cut; }
    bool hasAntennaDiffAreaRatio() { return _has_antenna_diff_area_ratio_pwl; }
    bool hasAntennaCumDiffAreaRatio() { return _has_antenna_cum_diff_area_ratio_pwl; }
    double get_antenna_area_ratio() { return _antenna_area_ratio; }
    double get_antenna_cum_area_ratio() { return _antenna_cum_area_ratio; }
    double get_anntenna_area_factory() { return _antenna_area_factory; }
    // setter
    void set_antenna_cum_routing_plus_cut(bool in) { _antenna_cum_routing_plus_cut = in; }
    void set_has_antenna_diff_area_ratio_pwl(bool in) { _has_antenna_diff_area_ratio_pwl = in; }
    void set_has_antenna_cum_diff_area_ratio_pwl(bool in) { _has_antenna_cum_diff_area_ratio_pwl = in; }
    void set_antenna_area_ratio(double in) { _antenna_area_ratio = in; }
    void set_antenna_cum_area_ratio(double in) { _antenna_cum_area_ratio = in; }
    void set_antenna_area_factory(double in) { _antenna_area_factory = in; }
    void add_antenna_cum_area_ratio_pwl(double diffusion, double ratio) { _antenna_diff_area_ratio_pwl[diffusion] = ratio; }
    void add_antenna_cum_diff_area_ratio_pwl(double diffusion, double ratio) {
      _antenna_cum_diff_area_ratio_pwl[diffusion] = ratio;
    }

    // other

   private:
    bool _antenna_cum_routing_plus_cut;
    bool _has_antenna_diff_area_ratio_pwl;
    bool _has_antenna_cum_diff_area_ratio_pwl;
    double _antenna_area_ratio;
    double _antenna_cum_area_ratio;
    double _antenna_area_factory = 1.0;

    std::map<double, double> _antenna_diff_area_ratio_pwl;
    std::map<double, double> _antenna_cum_diff_area_ratio_pwl;
  };

}  // namespace idb

#endif
