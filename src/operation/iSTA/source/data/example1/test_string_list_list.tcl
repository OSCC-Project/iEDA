# outputs:
#  < "a" "b" "c" >
#  < " o0 Oo O " "o!o@o" "" >
#  < "/a\\" "b,b" "dd" >
#  < "a_a-a" >
#  < "" >
#  < >
#  < "d" "dd" "ddd" >

test_string_list_list \
-puts {  { a b,c} ,,,,  ,{" o0 Oo O " o!o@o , ""}{/a\\, "b,b"  ,,, ,, dd  } {a_a-a},{""}{} } \
-puts { d dd ddd} 