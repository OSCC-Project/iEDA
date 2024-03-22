# Examples

## Example 1a [![wandbox.badge]](https://wandbox.org/permlink/4FF2LSXeDkXw1diP)
## Example 1b [![wandbox.badge]](https://wandbox.org/permlink/41riIYcZbYz8hjKG)
## Example 1c [![wandbox.badge]](https://wandbox.org/permlink/QE3g7YXo7whlv1fg)

Compile time matching only.

Does nothing at runtime.

## Example 2 [![wandbox.badge]](https://wandbox.org/permlink/jrmfdK9w4R8tBHPY)

Compile time and runtime check of string ID's validity using a pattern.

Defines `valid_id()` function which can be used when defining a string ID which
has to match a given pattern.

```C++
// compile time
constexpr auto id = valid_id("test_something", "[Tt]est_*");
```

When used at runtime and the string ID does not match,

```C++
// runtime
auto id = valid_id("tst_something_different", "[Tt]est_*");
```

`std::logic_error("Invalid ID")` is thrown.

**Usage:**

```batch
$ ./build/example/example02
Valid ID: test_something
Valid ID: Test_something_else
Valid ID: test_something
Valid ID: Test_something_else
Invalid ID
```

## Example 3 [![wandbox.badge]](https://wandbox.org/permlink/AX6LkvFR3wz4bPON)

Example 2 extended with a default pattern and with a user-defined string
literal.

The `valid_id()` can also be used with a default pattern

```C++
// compile time
constexpr auto id = valid_id("test_something");
```

and via `_valid_id` string literal.

```C++
// compile time
constexpr auto id = "test_something"_valid_id;
```

**Usage:**

```batch
$ ./build/example/example03
Valid ID: test_something
Valid ID: Test_something_else
Valid ID: test_something
Valid ID: Test_something_else
Invalid ID
```

## Example 4 [![wandbox.badge]](https://wandbox.org/permlink/5VoOQqA6DVmYKmlh)

Matching of command line arguments.

**Syntax:**

```batch
example04 [-v] sequence pattern
```

`-v` stands for verbose and can be used to visualize where exactly the matching
failed.

**Usage:**

```batch
$ ./build/example/example04 "You are so nice!" "[Y|y]ou are ((so |)nice|(too |)ugly)([.\!]|)"
$ echo $?
0
$
$ ./build/example/example04 "you're so nice" "[Y|y]ou are ((so |)nice|(too |)ugly)([.\!]|)"
$ echo $?
1
$
$ ./build/example/example04 -v "you're so nice" "[Y|y]ou are ((so |)nice|(too |)ugly)([.\!]|)"
you're so nice
   ^
[Y|y]ou are ((so |)nice|(too |)ugly)([.\!]|)
       ^
$ echo $?
1
```

## Example 5 [![wandbox.badge]](https://wandbox.org/permlink/NLY0d4zXZNP4Dv1A)

List files on a given path which match a given pattern.

This example makes use of `filesystem` library, therefore is not built for all
configurations.

**Syntax:**

```batch
example05 path pattern
```

**Usage:**

```batch
$ ./build/example/example05 test "*.[hc]pp"
test/src/catch.cpp
test/src/wildcards/match_test.cpp
test/src/cx/array_test.cpp
test/src/cx/tuple_test.cpp
test/src/cx/utility_test.cpp
test/src/cx/string_view_test.cpp
test/include/catch.hpp
```

## Example 6 [![wandbox.badge]](https://wandbox.org/permlink/Fwm46McV1qEciPdu)

Matching of generated DNA sequences.

A DNA sequence consists of letters `A`, `G`, `T`, `C` and is up to 10 letters
long (but can also be empty).

This example demonstrates the general purposeness of the library since the DNA
sequences are internally represented as shown below - i.e. not as strings.

```C++
enum class nucleobase
{
  adenine,
  cytosine,
  guanine,
  thymine
};

using dna_sequence_type = std::vector<nucleobase>;
```

**Syntax:**

```batch
example06 pattern
```

**Usage:**

```batch
$ ./build/example/example06 "?AG(TT|CC)*(A|G)"
...
...
...
TAGCCTAAG
TAGCCAG
GAGTTCG
CAGCCGG
TAGTTTCAA
GAGTTGG
CAGCCTG
GAGTTAA
TAGCCG
AAGTTGGTG
Generated : 1000000
Found     : 1528
```

[wandbox.badge]: https://img.shields.io/badge/try%20it-on%20wandbox-blue.svg
