use std::path::PathBuf;

use kcmt_core::telemetry::load_usage_summary;

use crate::args::StatsArgs;

pub fn render_stats_command(repo_path: PathBuf, args: StatsArgs) -> i32 {
    match load_usage_summary(&repo_path) {
        Ok(summary) if args.json => match serde_json::to_string_pretty(&summary) {
            Ok(rendered) => {
                println!("{rendered}");
                0
            }
            Err(err) => {
                eprintln!("{err}");
                1
            }
        },
        Ok(summary) => {
            println!("kcmt usage statistics");
            if summary.aggregates.is_empty() {
                println!("No usage telemetry recorded for this repository.");
                return 0;
            }
            println!("provider\tmodel\truns\tsuccesses\tfailures\tavg_latency_ms\trule");
            for aggregate in summary.aggregates {
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{:.1}\t{}",
                    aggregate.provider,
                    aggregate.model,
                    aggregate.runs,
                    aggregate.successes,
                    aggregate.failures,
                    aggregate.avg_latency_ms,
                    aggregate.selected_rule.unwrap_or_else(|| "-".to_string())
                );
            }
            0
        }
        Err(err) => {
            eprintln!("{err}");
            1
        }
    }
}
