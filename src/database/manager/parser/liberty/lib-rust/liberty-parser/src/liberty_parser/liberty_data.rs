trait LibertyAttrValue {
    fn is_string(&self) -> u32 {
        0
    }
    fn is_float(&self) -> u32 {
        0
    }

    fn get_float_value(&self) -> f64 {
        panic!("This is unknown value.");
    }
    fn get_string_value(&self) -> &str {
        panic!("This is unknown value.");
    }
}

struct LibertyFloatValue {
    value: f64,
}

impl LibertyAttrValue for LibertyFloatValue {
    fn is_float(&self) -> u32 {
        1
    }

    fn get_float_value(&self) -> f64 {
        self.value;
    }
}

struct LibertyStringValue {
    value: String,
}

impl LibertyAttrValue for LibertyStringValue {
    fn is_string(&self) -> u32 {
        1
    }

    fn get_string_value(&self) -> &str {
        &self.value
    }
}
