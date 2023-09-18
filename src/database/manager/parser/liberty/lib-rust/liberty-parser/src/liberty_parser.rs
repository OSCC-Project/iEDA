use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "liberty.pest"]
pub struct LibertyParser;
