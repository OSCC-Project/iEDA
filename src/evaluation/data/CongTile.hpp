#ifndef SRC_PLATFORM_EVALUATOR_DATA_CONGTILE_HPP_
#define SRC_PLATFORM_EVALUATOR_DATA_CONGTILE_HPP_

#include <memory>

namespace eval {

class Tile
{
 public:
  Tile() : _x(0), _y(0), _lx(0), _ly(0), _ux(0), _uy(0) {}
  Tile(int x, int y, int lx, int ly, int ux, int uy, int layer_index)
      : _x(x), _y(y), _lx(lx), _ly(ly), _ux(ux), _uy(uy), _layer_index(layer_index)
  {
  }
  ~Tile() { reset(); }

  // getter
  int get_x() const { return _x; }
  int get_y() const { return _y; }
  int get_lx() const { return _lx; }
  int get_ly() const { return _ly; }
  int get_ux() const { return _ux; }
  int get_uy() const { return _uy; }
  int get_east_cap() const { return _east_cap; }
  int get_east_use() const { return _east_use; }
  int get_west_cap() const { return _west_cap; }
  int get_west_use() const { return _west_use; }
  int get_south_cap() const { return _south_cap; }
  int get_south_use() const { return _south_use; }
  int get_north_cap() const { return _north_cap; }
  int get_north_use() const { return _north_use; }
  int get_track_cap() const { return _track_cap; }
  int get_track_use() const { return _track_use; }
  bool is_horizontal() const { return _is_horizontal; }

  void set_east_cap(const int& east_cap) { _east_cap = east_cap; }
  void set_east_use(const int& east_use) { _east_use = east_use; }
  void set_west_cap(const int& west_cap) { _west_cap = west_cap; }
  void set_west_use(const int& west_use) { _west_use = west_use; }
  void set_south_cap(const int& south_cap) { _south_cap = south_cap; }
  void set_south_use(const int& south_use) { _south_use = south_use; }
  void set_north_cap(const int& north_cap) { _north_cap = north_cap; }
  void set_north_use(const int& north_use) { _north_use = north_use; }
  void set_track_cap(const int& track_cap) { _track_cap = track_cap; }
  void set_track_use(const int& track_use) { _track_use = track_use; }
  void set_direction(const bool& is_horizontal) { _is_horizontal = is_horizontal; }

 private:
  int _x = 0;
  int _y = 0;
  int _lx = 0;
  int _ly = 0;
  int _ux = 0;
  int _uy = 0;

  // routing congestion
  int _east_cap = 0;
  int _east_use = 0;
  int _west_cap = 0;
  int _west_use = 0;
  int _south_cap = 0;
  int _south_use = 0;
  int _north_cap = 0;
  int _north_use = 0;
  int _track_cap = 0;
  int _track_use = 0;
  bool _is_horizontal = false;
  int _layer_index = 0;

  void reset();
};

class TileGrid
{
 public:
  TileGrid() : _lx(0), _ly(0), _tile_cnt_x(0), _tile_cnt_y(0), _tile_size_x(0), _tile_size_y(0), _num_routing_layers(0) {}
  TileGrid(int lx, int ly, int tileCntX, int tileCntY, int tileSizeX, int tileSizeY, int numRoutingLayers)
      : _lx(lx),
        _ly(ly),
        _tile_cnt_x(tileCntX),
        _tile_cnt_y(tileCntY),
        _tile_size_x(tileSizeX),
        _tile_size_y(tileSizeY),
        _num_routing_layers(numRoutingLayers)
  {
  }
  ~TileGrid() { reset(); }

  // getter
  int get_lx() const { return _lx; }
  int get_ly() const { return _ly; }
  int get_ux() const { return _lx + _tile_cnt_x * _tile_size_x; }
  int get_uy() const { return _ly + _tile_cnt_y * _tile_size_y; }
  int get_tile_cnt_x() const { return _tile_cnt_x; }
  int get_tile_cnt_y() const { return _tile_cnt_y; }
  int get_tile_size_x() const { return _tile_size_x; }
  int get_tile_size_y() const { return _tile_size_y; }
  int get_num_routing_layers() const { return _num_routing_layers; }
  std::vector<Tile*>& get_tiles() { return _tiles; }

  // setter
  void set_lx(const int& lx) { _lx = lx; }
  void set_ly(const int& ly) { _ly = ly; }
  void set_tile_cnt_x(const int& tileCntX) { _tile_cnt_x = tileCntX; }
  void set_tile_cnt_y(const int& tileCntY) { _tile_cnt_y = tileCntY; }
  void set_tileCnt(const int& tileCntX, const int& tileCntY)
  {
    _tile_cnt_x = tileCntX;
    _tile_cnt_y = tileCntY;
  }
  void set_tile_size_x(const int& tileSizeX) { _tile_size_x = tileSizeX; }
  void set_tile_size_y(const int& tileSizeY) { _tile_size_y = tileSizeY; }
  void set_tileSize(const int& tileSizeX, const int& tileSizeY) { _tile_size_x = tileSizeX, _tile_size_y = tileSizeY; }
  void set_num_routing_layers(const int& num) { _num_routing_layers = num; }

 private:
  int _lx;
  int _ly;
  int _tile_cnt_x;
  int _tile_cnt_y;
  int _tile_size_x;
  int _tile_size_y;
  int _num_routing_layers;

  std::vector<Tile*> _tiles;

  void reset();
};

inline void Tile::reset()
{
  _x = _y = _lx = _ly = _ux = _uy = 0;
  _east_cap = _east_use = _west_cap = _west_use = _south_cap = _south_use = _north_cap = _north_use = 0;
}

inline void TileGrid::reset()
{
  _lx = _ly = 0;
  _tile_cnt_x = _tile_cnt_y = 0;
  _tile_size_x = _tile_size_y = 0;
  _num_routing_layers = 0;

  _tiles.clear();
  _tiles.shrink_to_fit();
}

}  // namespace eval

#endif  // SRC_PLATFORM_EVALUATOR_DATA_CONGTILE_HPP_
