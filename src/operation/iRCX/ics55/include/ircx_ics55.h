#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(IRCX_ICS55_BUILD)
#define IRCX_ICS55_API __declspec(dllexport)
#else
#define IRCX_ICS55_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define IRCX_ICS55_API __attribute__((visibility("default")))
#else
#define IRCX_ICS55_API
#endif

// Returns non-zero on success and 0 on failure.
IRCX_ICS55_API int ircx_ics55_init(const char* config_file);
IRCX_ICS55_API int ircx_ics55_run(void);
IRCX_ICS55_API int ircx_ics55_run_with_idb_design(void* idb_design);
IRCX_ICS55_API int ircx_ics55_report(void);

// Convenience API: init + run + report.
IRCX_ICS55_API int ircx_ics55_extract(const char* config_file);
IRCX_ICS55_API int ircx_ics55_extract_with_idb_design(const char* config_file, void* idb_design);

// The returned pointer is owned by the library and remains valid until the next
// call from the same thread.
IRCX_ICS55_API const char* ircx_ics55_last_error(void);

#ifdef __cplusplus
}
#endif
