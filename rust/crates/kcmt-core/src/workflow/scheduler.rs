//! Parallel worker scheduling utilities for diff preparation.

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: String,
}

pub fn chunk_work(items: Vec<WorkItem>, workers: usize) -> Vec<Vec<WorkItem>> {
    let worker_count = workers.max(1);
    let mut buckets = vec![Vec::new(); worker_count];
    for (index, item) in items.into_iter().enumerate() {
        buckets[index % worker_count].push(item);
    }
    buckets
}
