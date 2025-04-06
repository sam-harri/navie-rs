// src/json_io.rs

use crate::domain::grid2d::Grid2D;
use crate::numerical::{interpolate_u_to_cell_centers, interpolate_v_to_cell_centers};
use serde::Serialize;
use serde_json;
use std::fs::{self, File};
use std::io::{self, Write, BufWriter};
use std::path::Path;
use tracing::info;
use std::time::Instant; // Import Instant

// --- Data Structures for Serialization ---
// (Keep Metadata, TimestepData, SimulationOutput as they were)
#[derive(Serialize, Debug)]
struct Metadata {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    re: f64,
    dt: f64,
    num_steps_completed: usize,
    output_frequency: Option<usize>,
}

#[derive(Serialize, Debug)]
pub struct TimestepData {
    pub step: usize,
    pub time: f64,
    pub u_centers: Vec<f64>,
    pub v_centers: Vec<f64>,
    pub p_centers: Vec<f64>,
}

#[derive(Serialize, Debug)]
struct SimulationOutput<'a> {
    pub metadata: Metadata,
    pub data: &'a [TimestepData],
}


// --- Output Manager ---
#[derive(Debug)]
pub struct JsonOutputManager {
    pub output_filepath: String,
    pub output_frequency: Option<usize>,
    pub collected_data: Vec<TimestepData>,
    pub nx: usize,
    pub ny: usize,
}

impl JsonOutputManager {
    pub fn new(
        output_filepath: String,
        output_frequency: Option<usize>,
        nx: usize,
        ny: usize,
        // dx: f64, // Pass dx/dy if needed for Metadata calculation later
        // dy: f64,
    ) -> Result<Self, io::Error> {
        let path = Path::new(&output_filepath);
        if let Some(parent_dir) = path.parent() {
            fs::create_dir_all(parent_dir)?;
            info!("Ensured output directory exists: {}", parent_dir.display());
        }

        Ok(Self {
            output_filepath,
            output_frequency,
            collected_data: Vec::new(),
            nx,
            ny,
            // dx, // Store if passed
            // dy,
        })
    }

    // should_collect method remains the same
    pub fn should_collect(&self, step: usize, _total_steps: usize, is_final_step: bool) -> bool {
         if let Some(freq) = self.output_frequency {
             if freq == 0 {
                 step == 0 || is_final_step
             } else {
                 step == 0 || step % freq == 0 || is_final_step
             }
         } else {
             true // Collect all if None freq
         }
     }


    // collect_timestep method remains the same
    pub fn collect_timestep(
         &mut self,
         step: usize,
         time: f64,
         grid: &Grid2D,
     ) -> Result<(), io::Error> {
         if grid.dimensions.0 != self.nx || grid.dimensions.1 != self.ny {
              return Err(io::Error::new(io::ErrorKind::InvalidInput,
                 format!("Grid dimensions ({},{}) do not match JsonOutputManager dimensions ({},{})",
                         grid.dimensions.0, grid.dimensions.1, self.nx, self.ny)));
         }
         info!("Collecting data for step {} (time {:.4})...", step, time);
         let u_centers_mat = interpolate_u_to_cell_centers(&grid.u);
         let v_centers_mat = interpolate_v_to_cell_centers(&grid.v);
         let p_centers_mat = &grid.pressure;
         let u_centers_vec = u_centers_mat.as_slice().to_vec();
         let v_centers_vec = v_centers_mat.as_slice().to_vec();
         let p_centers_vec = p_centers_mat.as_slice().to_vec();
         let data_point = TimestepData {
             step, time, u_centers: u_centers_vec, v_centers: v_centers_vec, p_centers: p_centers_vec,
         };
         self.collected_data.push(data_point);
         Ok(())
     }

    // write_final_output method remains the same
     pub fn write_final_output(
         &self,
         re: f64,
         dt: f64,
         dx: f64, // Pass dx, dy for metadata
         dy: f64,
         steps_completed: usize,
     ) -> Result<(), io::Error> {
         if self.collected_data.is_empty() {
             info!("No data collected, skipping JSON output to {}.", self.output_filepath);
             return Ok(());
         }
         info!("Writing collected data to JSON file: {}...", self.output_filepath);
         let output_start = Instant::now();
         let metadata = Metadata {
             nx: self.nx, ny: self.ny, dx, dy, // Use passed dx, dy
             re, dt, num_steps_completed: steps_completed, output_frequency: self.output_frequency,
         };
         let output_data = SimulationOutput { metadata, data: &self.collected_data };
         let json_string = match serde_json::to_string_pretty(&output_data) {
             Ok(s) => s,
             Err(e) => return Err(io::Error::new(io::ErrorKind::Other, format!("Failed to serialize data to JSON: {}", e))),
         };
         let file = File::create(&self.output_filepath)?;
         let mut writer = BufWriter::new(file);
         writer.write_all(json_string.as_bytes())?;
         writer.flush()?;
         info!("JSON output finished in {:.2}ms", output_start.elapsed().as_millis());
         Ok(())
     }
}

// --- Tests for json_io ---
// (Keep the tests as they were, they should work with public fields now)
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid2d::{GridDimensions2D, CellSize2D};
    use tempfile::tempdir;
    use std::fs;
    use approx::assert_relative_eq;

    fn create_dummy_grid(nx:usize, ny:usize) -> Grid2D {
        let dims = GridDimensions2D(nx, ny);
        let size = CellSize2D(1.0/nx as f64, 1.0/ny as f64);
        let mut grid = Grid2D::new(dims, size).unwrap();
        grid.u.fill(1.0); grid.v.fill(2.0); grid.pressure.fill(3.0);
        grid
    }

    #[test]
    fn test_manager_new_creates_dir() -> io::Result<()> {
        let dir = tempdir()?;
        let filepath = dir.path().join("subdir").join("output.json");
        let filepath_str = filepath.to_str().unwrap().to_string();
        assert!(!dir.path().join("subdir").exists());
        let _manager = JsonOutputManager::new(filepath_str, Some(10), 5, 5)?;
        assert!(dir.path().join("subdir").exists());
        dir.close()?;
        Ok(())
    }

     #[test]
    fn test_should_collect_logic() {
        let manager_freq10 = JsonOutputManager{ output_filepath: "".to_string(), output_frequency: Some(10), collected_data: vec![], nx: 10, ny: 10 };
        assert!(manager_freq10.should_collect(0, 100, false)); assert!(!manager_freq10.should_collect(1, 100, false)); assert!(manager_freq10.should_collect(10, 100, false)); assert!(manager_freq10.should_collect(99, 100, true)); assert!(manager_freq10.should_collect(100, 100, true));
        let manager_freq0 = JsonOutputManager{ output_filepath: "".to_string(), output_frequency: Some(0), collected_data: vec![], nx: 10, ny: 10 };
        assert!(manager_freq0.should_collect(0, 100, false)); assert!(!manager_freq0.should_collect(1, 100, false)); assert!(manager_freq0.should_collect(100, 100, true));
        let manager_freq_none = JsonOutputManager{ output_filepath: "".to_string(), output_frequency: None, collected_data: vec![], nx: 10, ny: 10 };
        assert!(manager_freq_none.should_collect(0, 100, false)); assert!(manager_freq_none.should_collect(1, 100, false)); assert!(manager_freq_none.should_collect(100, 100, true));
    }

    #[test]
    fn test_collect_and_write() -> io::Result<()> {
        let nx = 3; let ny = 2; let dx = 1.0/3.0; let dy = 0.5;
        let dir = tempdir()?;
        let filepath = dir.path().join("collect_test.json");
        let filepath_str = filepath.to_str().unwrap().to_string();
        let mut manager = JsonOutputManager::new(filepath_str.clone(), None, nx, ny)?;
        let grid1 = create_dummy_grid(nx, ny);
        manager.collect_timestep(0, 0.0, &grid1)?;
        let mut grid2 = create_dummy_grid(nx, ny); grid2.pressure.fill(5.0);
        manager.collect_timestep(1, 0.1, &grid2)?;
        assert_eq!(manager.collected_data.len(), 2);
        manager.write_final_output(100.0, 0.1, dx, dy, 1)?; // Pass dx, dy
        let content = fs::read_to_string(filepath)?;
        let output: serde_json::Value = serde_json::from_str(&content)?;
        assert_eq!(output["metadata"]["nx"], nx); assert_eq!(output["metadata"]["re"], 100.0); assert_eq!(output["data"].as_array().unwrap().len(), 2);
        let p1_flat = output["data"][1]["p_centers"].as_array().unwrap(); assert_relative_eq!(p1_flat[0].as_f64().unwrap(), 5.0);
        dir.close()?;
        Ok(())
    }
}