#[derive(Clone, Debug)]
enum ConnectionType {
    INTERNAL,
    EXTERNAL,
    UNITIALIZED,
}

#[derive(Clone, Debug)]
enum ConnectionDirection {
    INPUT,
    OUTPUT,
    INOUT,
    UNITIALIZED,
}

#[derive(Clone, Debug)]
struct HeaderItem {
    key: String,
    value: String,
}

#[derive(Clone, Debug)]
struct NameMapItem {
    index: usize,
    name: String,
}

#[derive(Clone, Debug)]
struct PortItem {
    name: String,
    direction: ConnectionDirection,
    coordinates: [f64; 2],
}

#[derive(Clone, Debug)]
struct ConnItem {
    conn_type: ConnectionType,
    conn_direction: ConnectionDirection,
    pin_name: String,
    driving_cell: String,
    load: f64,
    layer: usize,
    coordinates: [f64; 2],
    ll_coordinate: [f64; 2],
    ur_coordinate: [f64; 2],
}

#[derive(Clone, Debug)]
struct CapItem {
    pin_port: [String; 2],
    cap_val: f64,
}

#[derive(Clone, Debug)]
struct ResItem {
    pin_port: [String; 2],
    res: f64,
}

#[derive(Clone, Debug)]
struct NetItem {
    name: String,
    lcap: f64,
    conns: Vec<ConnItem>,
    caps: Vec<CapItem>,
    ress: Vec<ResItem>,
}
// Shared structure between cpp and rust, it uses the types in the `exter "Rust" section`
#[derive(Clone, Debug)]
struct SpefFile {
    name: String,
    header: Vec<HeaderItem>,
    name_vector: Vec<NameMapItem>,
    ports: Vec<PortItem>,
    nets: Vec<NetItem>,
}
