#ifndef IDB_MIN_WIDTH_CHECK
#define IDB_MIN_WIDTH_CHECK

namespace idb {
  class IdbMinWidthCheck {
   public:
    IdbMinWidthCheck() { }
    explicit IdbMinWidthCheck(int minWidth) : _min_width(minWidth) { }
    ~IdbMinWidthCheck() = default;

    // getter
    int get_min_width() const { return _min_width; }
    // setter
    void set_min_width(int min_width) { _min_width = min_width; }
    // operator

   private:
    int _min_width;
  };
}  // namespace idb

#endif
