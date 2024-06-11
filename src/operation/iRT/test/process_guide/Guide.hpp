class Guide
{
 public:
  Guide() = default;
  ~Guide() = default;
  // getter
  int32_t get_ll_x() const { return _ll_x; }
  int32_t get_ll_y() const { return _ll_y; }
  int32_t get_ur_x() const { return _ur_x; }
  int32_t get_ur_y() const { return _ur_y; }
  std::string& get_layer_name() { return _layer_name; }
  // setter
  void set_ll_x(const int32_t ll_x) { _ll_x = ll_x; }
  void set_ll_y(const int32_t ll_y) { _ll_y = ll_y; }
  void set_ur_x(const int32_t ur_x) { _ur_x = ur_x; }
  void set_ur_y(const int32_t ur_y) { _ur_y = ur_y; }
  void set_layer_name(const std::string& layer_name) { _layer_name = layer_name; }
  // function

 private:
  int32_t _ll_x;
  int32_t _ll_y;
  int32_t _ur_x;
  int32_t _ur_y;
  std::string _layer_name;
};