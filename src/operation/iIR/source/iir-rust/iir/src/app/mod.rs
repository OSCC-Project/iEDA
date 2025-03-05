use std::io::Write;

pub fn init_ir() {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{timestamp}][{file_name}:{line_no}][{thread_id:?}][{level}] {args}",
                timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                file_name = record.file_static().unwrap(),
                line_no = record.line().unwrap(),
                thread_id = std::thread::current().id(),
                level = record.level(),
                args = record.args(),
            )
        })
        .init();
}

#[no_mangle]
pub extern "C" fn init_iir() {
    init_ir();
}
