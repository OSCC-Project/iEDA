# Changelog for scipoptsuite-8.0.3 

## Reason

Due to the existence of a certain 'assert' instruction that forcefully interrupts the solving process, the code has been temporarily commented out.

## Diff

```
./soplex/src/soplex/spxsolve.hpp
<                // assert(EQrel(value(), origval, alloweddeviation));
---
>                assert(EQrel(value(), origval, alloweddeviation));
```