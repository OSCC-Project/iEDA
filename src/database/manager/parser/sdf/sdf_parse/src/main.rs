pub mod parse_sdf;
pub mod test_parse_fun;
fn main() {
    test_parse_fun::test_sdf_header();
    test_parse_fun::test_del_spec();
    test_parse_fun::test_tc_spec();
    test_parse_fun::test_te_spec();
    
    test_parse_fun::test_parse_sdf();
    

}
