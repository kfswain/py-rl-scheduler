mod types;
mod interface;
mod scheduler;

use pyo3::prelude::*;

#[pymodule]
fn _scheduling(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<types::LLMRequest>()?;
    m.add_class::<types::Endpoint>()?;
    m.add_class::<types::ScoredEndpoint>()?;
    m.add_class::<types::ProfileRunResult>()?;
    m.add_class::<types::SchedulingResult>()?;
    m.add_class::<types::CycleState>()?;

    m.add_class::<interface::WeightedScorer>()?;
    m.add_class::<interface::SchedulerProfile>()?;

    m.add_class::<scheduler::RustSchedulerConfig>()?;
    m.add_class::<scheduler::Scheduler>()?;

    Ok(())
}
