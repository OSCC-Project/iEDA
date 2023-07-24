puts "###### test_args ######"

puts "argc = $argc"

set i 0
foreach arg $argv {
    puts "arg $i: $arg"
    incr i
}

puts "###### test_args ######"