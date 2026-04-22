use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(get_all, set_all, subclass)]
pub struct LLMRequest {
    pub request_id: String,
    pub target_model: Option<String>,
    pub headers: HashMap<String, String>,
    pub body: PyObject,
}

impl Clone for LLMRequest {
    fn clone(&self) -> Self {
        Python::with_gil(|py| LLMRequest {
            request_id: self.request_id.clone(),
            target_model: self.target_model.clone(),
            headers: self.headers.clone(),
            body: self.body.clone_ref(py),
        })
    }
}

#[pymethods]
impl LLMRequest {
    #[new]
    #[pyo3(signature = (request_id, target_model=None, headers=None, body=None))]
    fn new(
        request_id: String,
        target_model: Option<String>,
        headers: Option<HashMap<String, String>>,
        body: Option<PyObject>,
    ) -> Self {
        LLMRequest {
            request_id,
            target_model,
            headers: headers.unwrap_or_default(),
            body: body.unwrap_or_else(|| Python::with_gil(|py| py.None())),
        }
    }
}

#[pyclass(get_all, set_all, subclass)]
pub struct Endpoint {
    pub name: String,
    pub attributes: HashMap<String, PyObject>,
}

impl Clone for Endpoint {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Endpoint {
            name: self.name.clone(),
            attributes: self.attributes.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect(),
        })
    }
}

#[pymethods]
impl Endpoint {
    #[new]
    #[pyo3(signature = (name, attributes=None))]
    fn new(name: String, attributes: Option<HashMap<String, PyObject>>) -> Self {
        Endpoint {
            name,
            attributes: attributes.unwrap_or_default(),
        }
    }
}

#[pyclass(get_all, set_all, subclass)]
#[derive(Clone)]
pub struct ScoredEndpoint {
    pub endpoint: Endpoint,
    pub score: f64,
}

#[pymethods]
impl ScoredEndpoint {
    #[new]
    fn new(endpoint: Endpoint, score: f64) -> Self {
        ScoredEndpoint { endpoint, score }
    }
}

#[pyclass(get_all, set_all, subclass)]
#[derive(Default)]
pub struct ProfileRunResult {
    pub endpoint_list: Vec<ScoredEndpoint>,
    pub metadata: HashMap<String, PyObject>,
}

impl Clone for ProfileRunResult {
    fn clone(&self) -> Self {
        Python::with_gil(|py| ProfileRunResult {
            endpoint_list: self.endpoint_list.clone(),
            metadata: self.metadata.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect(),
        })
    }
}

#[pymethods]
impl ProfileRunResult {
    #[new]
    #[pyo3(signature = (endpoint_list=None, metadata=None))]
    fn new(
        endpoint_list: Option<Vec<ScoredEndpoint>>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> Self {
        ProfileRunResult {
            endpoint_list: endpoint_list.unwrap_or_default(),
            metadata: metadata.unwrap_or_default(),
        }
    }
}

#[pyclass(get_all, set_all, subclass)]
pub struct SchedulingResult {
    pub profile_results: HashMap<String, ProfileRunResult>,
    pub primary_profile_name: Option<String>,
}

#[pymethods]
impl SchedulingResult {
    #[new]
    #[pyo3(signature = (profile_results=None, primary_profile_name=None))]
    fn new(
        profile_results: Option<HashMap<String, ProfileRunResult>>,
        primary_profile_name: Option<String>,
    ) -> Self {
        SchedulingResult {
            profile_results: profile_results.unwrap_or_default(),
            primary_profile_name,
        }
    }
}

#[pyclass(subclass)]
pub struct CycleState {
    pub _state: HashMap<String, PyObject>,
}

#[pymethods]
impl CycleState {
    #[new]
    pub fn new() -> Self {
        CycleState {
            _state: HashMap::new(),
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyObject {
        self._state
            .get(key)
            .map(|v| v.clone_ref(py))
            .unwrap_or_else(|| default.unwrap_or_else(|| py.None()))
    }

    fn set(&mut self, key: String, value: PyObject) {
        self._state.insert(key, value);
    }
}
