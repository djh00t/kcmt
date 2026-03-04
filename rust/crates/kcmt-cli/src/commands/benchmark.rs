use kcmt_bench::export::{export_csv, export_json};
use kcmt_bench::runner::run_benchmark;

pub fn run_benchmark_command() -> i32 {
    let run = run_benchmark(".");
    match export_json(&run) {
        Ok(json) => {
            println!("{json}");
            let _ = export_csv(&run);
            0
        }
        Err(err) => {
            eprintln!("benchmark export failed: {err}");
            1
        }
    }
}
