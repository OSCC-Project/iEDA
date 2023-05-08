#ifndef SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALLAYOUT_HPP_
#define SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALLAYOUT_HPP_

#include <cmath>

#include "CongBin.hpp"
#include "CongTile.hpp"
#include "EvalRect.hpp"

namespace eval {

class Layout
{
 public:
  Layout() : _tile_grid(new TileGrid()), _cong_grid(new CongGrid()) {}
  Layout(const Layout&) = delete;
  Layout(Layout&&) = delete;
  ~Layout()
  {
    delete _tile_grid;
    delete _cong_grid;
  }

  Layout& operator=(const Layout&) = delete;
  Layout& operator=(Layout&&) = delete;

  // getter.
  int32_t get_database_unit() const { return _database_unit; }

  Rectangle<int32_t> get_die_shape() const { return _die_shape; }
  int32_t get_die_width() { return _die_shape.get_width(); }
  int32_t get_die_height() { return _die_shape.get_height(); }

  Rectangle<int32_t> get_core_shape() const { return _core_shape; }
  int32_t get_core_width() { return _core_shape.get_width(); }
  int32_t get_core_height() { return _core_shape.get_height(); }

  TileGrid* get_tile_grid() const { return _tile_grid; }
  CongGrid* get_cong_grid() const { return _cong_grid; }

  // setter.
  void set_database_unit(int32_t dbu) { _database_unit = dbu; }
  void set_die_shape(Rectangle<int32_t> rect) { _die_shape = std::move(rect); }
  void set_core_shape(Rectangle<int32_t> rect) { _core_shape = std::move(rect); }

  void set_tile_grid(int32_t tile_size_x, int32_t tile_size_y, int32_t num_routing_layers);
  void set_cong_grid(int32_t bin_cnt_x, int32_t bin_cnt_y);
  void set_cong_grid(int32_t bin_cnt_x, int32_t bin_cnt_y, idb::IdbLayers* idb_layers);
  void set_cong_grid_for_predictor(int32_t tile_size_x, int32_t tile_size_y, idb::IdbLayers* idb_layers);

 private:
  int32_t _database_unit;
  Rectangle<int32_t> _die_shape;
  Rectangle<int32_t> _core_shape;
  TileGrid* _tile_grid;
  CongGrid* _cong_grid;
};

inline void Layout::set_tile_grid(int32_t tile_size_x, int32_t tile_size_y, int32_t num_routing_layers)
{
  _tile_grid->set_lx(_die_shape.get_ll_x());
  _tile_grid->set_ly(_die_shape.get_ll_y());
  _tile_grid->set_tile_cnt_x(ceil(_die_shape.get_width() / (float) tile_size_x));
  _tile_grid->set_tile_cnt_y(ceil(_die_shape.get_height() / (float) tile_size_y));
  _tile_grid->set_tile_size_x(tile_size_x);
  _tile_grid->set_tile_size_y(tile_size_y);
  _tile_grid->set_num_routing_layers(num_routing_layers);
  // _tile_grid->initTiles();
}

inline void Layout::set_cong_grid(int32_t bin_cnt_x, int32_t bin_cnt_y)
{
  _cong_grid->set_lx(_core_shape.get_ll_x());
  _cong_grid->set_ly(_core_shape.get_ll_y());
  _cong_grid->set_bin_cnt_x(bin_cnt_x);
  _cong_grid->set_bin_cnt_y(bin_cnt_y);
  _cong_grid->set_bin_size_x(ceil(_core_shape.get_width() / (float) bin_cnt_x));
  _cong_grid->set_bin_size_y(ceil(_core_shape.get_height() / (float) bin_cnt_y));
  _cong_grid->initBins();
}

inline void Layout::set_cong_grid(int32_t bin_cnt_x, int32_t bin_cnt_y, idb::IdbLayers* idb_layers)
{
  _cong_grid->set_lx(_core_shape.get_ll_x());
  _cong_grid->set_ly(_core_shape.get_ll_y());
  _cong_grid->set_bin_cnt_x(bin_cnt_x);
  _cong_grid->set_bin_cnt_y(bin_cnt_y);
  _cong_grid->set_bin_size_x(ceil(_core_shape.get_width() / (float) bin_cnt_x));
  _cong_grid->set_bin_size_y(ceil(_core_shape.get_height() / (float) bin_cnt_y));
  _cong_grid->set_routing_layers_number(idb_layers->get_routing_layers_number());
  _cong_grid->initBins(idb_layers);
}

inline void Layout::set_cong_grid_for_predictor(int32_t tile_size_x, int32_t tile_size_y, idb::IdbLayers* idb_layers)
{
  _cong_grid->set_lx(_core_shape.get_ll_x());
  _cong_grid->set_ly(_core_shape.get_ll_y());
  _cong_grid->set_bin_cnt_x(ceil(_core_shape.get_width() / (float) tile_size_x));
  _cong_grid->set_bin_cnt_y(ceil(_core_shape.get_height() / (float) tile_size_y));
  _cong_grid->set_bin_size_x(tile_size_x);
  _cong_grid->set_bin_size_y(tile_size_y);
  _cong_grid->set_routing_layers_number(idb_layers->get_routing_layers_number());
  _cong_grid->initBins(idb_layers);
}

}  // namespace eval

#endif  // SRC_EVALUATOR_SOURCE_WRAPPER_DATABASE_EVALLAYOUT_HPP_
