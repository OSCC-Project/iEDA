#ifndef IDB_MIN_AREA_CHECK
#define IDB_MIN_AREA_CHECK

namespace idb {
  class IdbMinAreaCheck {
   public:
    IdbMinAreaCheck() { }
    explicit IdbMinAreaCheck(int min_area) : _min_area(min_area) { }
    ~IdbMinAreaCheck() = default;
    // getter
    int get_min_area() const { return _min_area; }
    // setter
    void set_min_area(int min_area) { _min_area = min_area; }
    // operator

   private:
    int _min_area;
  };
}  // namespace idb

#endif
