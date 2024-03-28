# How to

## How to get the sources

```batch
$ git clone https://github.com/zemasoft/wildcards
$ cd wildcards
```

## How to build tests

```batch
$ cmake . -Bbuild -DCMAKE_BUILD_TYPE=Debug -DWILDCARDS_BUILD_TESTS=ON
$ cmake --build build
```

## How to run the tests

```batch
$ cd build
$ ctest
```

## How to build examples

```batch
$ cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DWILDCARDS_BUILD_EXAMPLES=ON
$ cmake --build build
```

## How to find the examples

```batch
$ ls build/example/example*
```

## How to build all

```batch
$ cmake . -Bbuild -DCMAKE_BUILD_TYPE=Debug -DWILDCARDS_BUILD_TESTS=ON -DWILDCARDS_BUILD_EXAMPLES=ON
$ cmake --build build
```
