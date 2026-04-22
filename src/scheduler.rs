use crate::interface::SchedulerProfile;
use crate::types::{CycleState, Endpoint, LLMRequest, ProfileRunResult, SchedulingResult, ScoredEndpoint};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

#[pyclass(name = "RustSchedulerConfig", get_all, set_all, subclass)]
pub struct RustSchedulerConfig {
    pub profile_handler: PyObject,
    pub profiles: HashMap<String, Py<SchedulerProfile>>,
}

impl Clone for RustSchedulerConfig {
    fn clone(&self) -> Self {
        Python::with_gil(|py| RustSchedulerConfig {
            profile_handler: self.profile_handler.clone_ref(py),
            profiles: self.profiles.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect(),
        })
    }
}

#[pymethods]
impl RustSchedulerConfig {
    #[new]
    fn new(profile_handler: PyObject, profiles: HashMap<String, Py<SchedulerProfile>>) -> Self {
        RustSchedulerConfig {
            profile_handler,
            profiles,
        }
    }
}

#[pyclass(subclass)]
pub struct Scheduler {
    pub config_path: Option<PathBuf>,
    pub last_mtime: Option<SystemTime>,
    pub profile_handler: Option<PyObject>,
    pub profiles: HashMap<String, Py<SchedulerProfile>>,
}

#[pymethods]
impl Scheduler {
    #[new]
    #[pyo3(signature = (config_path=None))]
    fn new(config_path: Option<String>) -> PyResult<Self> {
        let path = config_path.map(PathBuf::from).or_else(|| {
            std::env::var("ROUTER_CONFIG_PATH").ok().map(PathBuf::from)
        });

        let mut scheduler = Scheduler {
            config_path: path,
            last_mtime: None,
            profile_handler: None,
            profiles: HashMap::new(),
        };

        scheduler._maybe_reload_config()?;
        Ok(scheduler)
    }

    #[staticmethod]
    fn new_with_config(config: &RustSchedulerConfig) -> Self {
        Python::with_gil(|py| Scheduler {
            config_path: None,
            last_mtime: None,
            profile_handler: Some(config.profile_handler.clone_ref(py)),
            profiles: config.profiles.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect(),
        })
    }

    fn _maybe_reload_config(&mut self) -> PyResult<()> {
        let path = match &self.config_path {
            Some(p) => p,
            None => return Ok(()),
        };

        let metadata = fs::metadata(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to get metadata for {:?}: {}", path, e))
        })?;
        let mtime = metadata.modified().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to get mtime for {:?}: {}", path, e))
        })?;

        if self.last_mtime.is_none() || mtime > self.last_mtime.unwrap() {
            println!("Reloading scheduler config from {:?}", path);
            Python::with_gil(|py| {
                let yaml = py.import("yaml")?;
                let builtins = py.import("builtins")?;
                let file = builtins.call_method1("open", (path.to_str().unwrap(), "r"))?;
                let config_dict = yaml.call_method1("safe_load", (file,))?;
                
                let core_config = py.import("scheduling.core.config")?;
                let scheduler_config_cls = core_config.getattr("SchedulerConfig")?;
                let config: Bound<'_, PyAny> = scheduler_config_cls.call_method1("from_dict", (config_dict,))?;
                
                let rust_config: PyRef<'_, RustSchedulerConfig> = config.extract()?;
                
                self.profile_handler = Some(rust_config.profile_handler.clone_ref(py));
                self.profiles = rust_config.profiles.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect();
                self.last_mtime = Some(mtime);
                Ok(())
            })
        } else {
            Ok(())
        }
    }

    fn schedule(
        &mut self,
        py: Python<'_>,
        request: LLMRequest,
        candidates: Vec<Endpoint>,
    ) -> PyResult<SchedulingResult> {
        if candidates.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("no scheduling candidates provided"));
        }

        let cycle_state = Py::new(py, CycleState::new())?;
        let mut profile_results: HashMap<String, Option<ProfileRunResult>> = HashMap::new();

        let handler_obj = self.profile_handler.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Profile handler not initialized")
        })?;
        let handler = handler_obj.bind(py);

        // Cloned profiles map for Python call
        let profiles_cloned: HashMap<String, Py<SchedulerProfile>> = self.profiles.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect();
        let profile_results_cloned: HashMap<String, Option<ProfileRunResult>> = profile_results.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // ask profile handler which profiles to run
        let selected: HashMap<String, Py<SchedulerProfile>> = handler
            .call_method1("pick", (cycle_state.clone_ref(py), request.clone(), profiles_cloned, profile_results_cloned))?
            .extract()?;

        for (name, profile_py) in selected {
            let profile = profile_py.borrow(py);
            let res = profile.run(py, request.clone(), cycle_state.clone_ref(py), candidates.clone());
            match res {
                Ok(r) => {
                    profile_results.insert(name, Some(r));
                }
                Err(e) => {
                    println!("Error running profile {}: {:?}", name, e);
                    profile_results.insert(name, None);
                }
            }
        }

        let profile_results_cloned_2: HashMap<String, Option<ProfileRunResult>> = profile_results.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let primary: Option<String> = handler
            .call_method1("process_results", (cycle_state.clone_ref(py), request.clone(), profile_results_cloned_2))?
            .extract()?;

        let primary_name = primary.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No primary profile selected")
        })?;

        // Build SchedulingResult
        let mut result_profile_results = HashMap::new();
        for (k, v) in profile_results {
            result_profile_results.insert(k, v.unwrap_or_default());
        }

        let result = SchedulingResult {
            profile_results: result_profile_results,
            primary_profile_name: Some(primary_name.clone()),
        };

        if let Some(profile_res) = result.profile_results.get(&primary_name) {
            if let Some(selected_ep) = profile_res.endpoint_list.first() {
                if let Some(profile_py) = self.profiles.get(&primary_name) {
                    let profile = profile_py.borrow(py);
                    for w in &profile.scorers {
                        let scorer = w.scorer.bind(py);
                        if scorer.getattr("pre_request").is_ok() {
                            scorer.call_method1("pre_request", (cycle_state.clone_ref(py), request.clone(), selected_ep.endpoint.clone()))?;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    fn run(
        &mut self,
        py: Python<'_>,
        request: LLMRequest,
        candidates: Vec<Endpoint>,
    ) -> PyResult<Vec<ScoredEndpoint>> {
        self._maybe_reload_config()?;
        let scheduler_output = self.schedule(py, request, candidates)?;
        let profile_name = scheduler_output.primary_profile_name.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Primary profile name is None")
        })?;
        
        if let Some(profile_results) = scheduler_output.profile_results.get(profile_name) {
             if let Some(selected_endpoint) = profile_results.endpoint_list.first() {
                 return Ok(vec![selected_endpoint.clone()]);
             }
        }

        println!("No endpoint selected, defaulting to framework routing logic");
        Ok(vec![])
    }
}
