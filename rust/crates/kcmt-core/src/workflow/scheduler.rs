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

#[cfg(test)]
mod tests {
    use super::{chunk_work, WorkItem};

    fn ids(buckets: &[Vec<WorkItem>]) -> Vec<Vec<String>> {
        buckets
            .iter()
            .map(|bucket| bucket.iter().map(|item| item.id.clone()).collect())
            .collect()
    }

    #[test]
    fn chunk_work_uses_one_bucket_when_workers_zero() {
        let buckets = chunk_work(
            vec![WorkItem { id: "a".into() }, WorkItem { id: "b".into() }],
            0,
        );

        assert_eq!(ids(&buckets), vec![vec!["a".to_string(), "b".to_string()]]);
    }

    #[test]
    fn chunk_work_distributes_round_robin_across_workers() {
        let buckets = chunk_work(
            vec![
                WorkItem { id: "a".into() },
                WorkItem { id: "b".into() },
                WorkItem { id: "c".into() },
                WorkItem { id: "d".into() },
                WorkItem { id: "e".into() },
            ],
            3,
        );

        assert_eq!(
            ids(&buckets),
            vec![
                vec!["a".to_string(), "d".to_string()],
                vec!["b".to_string(), "e".to_string()],
                vec!["c".to_string()],
            ]
        );
    }

    #[test]
    fn chunk_work_preserves_empty_extra_buckets() {
        let buckets = chunk_work(vec![WorkItem { id: "only".into() }], 3);

        assert_eq!(
            ids(&buckets),
            vec![vec!["only".to_string()], vec![], vec![]]
        );
    }
}
