include(FetchContent)
FetchContent_Populate(
  tbb
  URL https://github.com/oneapi-src/oneTBB/releases/download/v2021.7.0/oneapi-tbb-2021.7.0-lin.tgz
  URL_HASH SHA256=3c2b3287c595e2bb833c025fcd271783963b7dfae8dc681440ea6afe5d550e6a
  SOURCE_DIR external_tools/tbb
)