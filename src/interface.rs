use crate::types::{
    CycleState, Endpoint, LLMRequest, ProfileRunResult, ScoredEndpoint,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

#[pyclass(get_all, set_all, subclass)]
pub struct WeightedScorer {
    pub scorer: PyObject,
    pub weight: f64,
}

impl Clone for WeightedScorer {
    fn clone(&self) -> Self {
        Python::with_gil(|py| WeightedScorer {
            scorer: self.scorer.clone_ref(py),
            weight: self.weight,
        })
    }
}

#[pymethods]
impl WeightedScorer {
    #[new]
    #[pyo3(signature = (scorer, weight=1.0))]
    fn new(scorer: PyObject, weight: f64) -> Self {
        WeightedScorer { scorer, weight }
    }
}

#[pyclass(get_all, set_all, subclass)]
pub struct SchedulerProfile {
    pub name: String,
    pub filters: Vec<PyObject>,
    pub scorers: Vec<WeightedScorer>,
    pub picker: Option<PyObject>,
}

#[pymethods]
impl SchedulerProfile {
    #[new]
    fn new(name: String) -> Self {
        SchedulerProfile {
            name,
            filters: Vec::new(),
            scorers: Vec::new(),
            picker: None,
        }
    }

    #[pyo3(signature = (*filters))]
    fn with_filters(slf: PyRefMut<'_, Self>, filters: &Bound<'_, PyTuple>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut slf_mut = slf;
        for f in filters.iter() {
            slf_mut.filters.push(f.to_object(py));
        }
        let res = slf_mut.into_py(py);
        Ok(res.extract::<Py<SchedulerProfile>>(py)?)
    }

    #[pyo3(signature = (*scorers))]
    fn with_scorers(slf: PyRefMut<'_, Self>, scorers: &Bound<'_, PyTuple>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut slf_mut = slf;
        for s in scorers.iter() {
            let weighted: WeightedScorer = s.extract()?;
            slf_mut.scorers.push(weighted);
        }
        let res = slf_mut.into_py(py);
        Ok(res.extract::<Py<SchedulerProfile>>(py)?)
    }

    fn with_picker(slf: PyRefMut<'_, Self>, picker: PyObject) -> PyResult<Py<Self>> {
        let py = slf.py();
        let mut slf_mut = slf;
        slf_mut.picker = Some(picker);
        let res = slf_mut.into_py(py);
        Ok(res.extract::<Py<SchedulerProfile>>(py)?)
    }

    pub fn run(
        &self,
        py: Python<'_>,
        request: LLMRequest,
        cycle_state: Py<CycleState>,
        candidates: Vec<Endpoint>,
    ) -> PyResult<ProfileRunResult> {
        let mut endpoints_map: HashMap<String, Endpoint> = candidates
            .into_iter()
            .map(|e| (e.name.clone(), e))
            .collect();

        // run filters
        for filter_plugin in &self.filters {
            let dict = PyDict::new(py);
            for (k, v) in &endpoints_map {
                dict.set_item(k, v.clone().into_py_any(py)?)?;
            }
            let filtered_map: HashMap<String, Endpoint> = filter_plugin
                .call_method1(py, "filter", (cycle_state.clone_ref(py), request.clone(), dict))?
                .extract(py)?;
            endpoints_map = filtered_map;
        }

        if endpoints_map.is_empty() {
            return Ok(ProfileRunResult::default());
        }

        // combine scores
        let mut total_scores: HashMap<String, f64> = endpoints_map.keys().map(|k| (k.clone(), 0.0)).collect();

        for w in &self.scorers {
            let dict = PyDict::new(py);
            for (k, v) in &endpoints_map {
                dict.set_item(k, v.clone().into_py_any(py)?)?;
            }
            let raw_scores: HashMap<String, f64> = w.scorer
                .call_method1(py, "score", (cycle_state.clone_ref(py), request.clone(), dict))?
                .extract(py)?;

            let vals: Vec<f64> = endpoints_map.keys().map(|k| *raw_scores.get(k).unwrap_or(&0.0)).collect();

            if !vals.is_empty() {
                let mut min_v = vals[0];
                let mut max_v = vals[0];
                for &v in &vals {
                    if v < min_v { min_v = v; }
                    if v > max_v { max_v = v; }
                }

                if max_v > min_v {
                    for (name, _) in &endpoints_map {
                        let v = *raw_scores.get(name).unwrap_or(&0.0);
                        let norm = (v - min_v) / (max_v - min_v);
                        *total_scores.get_mut(name).unwrap() += norm * w.weight;
                    }
                } else {
                    for (name, _) in &endpoints_map {
                        *total_scores.get_mut(name).unwrap() += 1.0 * w.weight;
                    }
                }
            }
        }

        let mut scored: Vec<ScoredEndpoint> = endpoints_map.into_iter().map(|(name, endpoint)| {
            ScoredEndpoint {
                endpoint,
                score: *total_scores.get(&name).unwrap_or(&0.0),
            }
        }).collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let chosen = if let Some(picker) = &self.picker {
            let picked: Option<ScoredEndpoint> = picker
                .call_method1(py, "pick", (cycle_state, request, scored))?
                .extract(py)?;
            picked.into_iter().collect()
        } else {
            scored.first().cloned().into_iter().collect()
        };

        Ok(ProfileRunResult {
            endpoint_list: chosen,
            metadata: HashMap::new(),
        })
    }
}
