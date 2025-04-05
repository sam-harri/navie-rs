#![allow(unused)]

use crate::domain::grid2d::Grid2D; // Needed if passing Grid2D directly (alternative: pass fields)
use crate::numerical::{interpolate_u_to_cell_centers, interpolate_v_to_cell_centers}; // For interpolation
use serde::Serialize;
use serde_json;
use std::fs::{self, File};
use std::io::{self, Write, BufWriter};
use std::path::Path;
use tracing::info; // For logging within this module

// --- Data Structures for Serialization ---

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
struct TimestepData {
    step: usize,
    time: f64,
    // Store fields as flattened Vec<f64> (column-major)
    u_centers: Vec<f64>,
    v_centers: Vec<f64>,
    p_centers: Vec<f64>,
}

#[derive(Serialize, Debug)]
struct SimulationOutput<'a> { // Use lifetime for borrowing data
    metadata: Metadata,
    data: &'a [TimestepData], // Borrow the collected data
}

// --- Output Manager ---

#[derive(Debug)]
pub struct JsonOutputManager {
    output_filepath: String, // Filepath is mandatory for this manager
    output_frequency: Option<usize>,
    collected_data: Vec<TimestepData>,
    // Store grid dimensions needed for interpolation/flattening
    nx: usize,
    ny: usize,
}

impl JsonOutputManager {
    /// Creates a new JsonOutputManager.
    /// Ensures the parent directory for the output file exists.
    pub fn new(
        output_filepath: String,
        output_frequency: Option<usize>,
        nx: usize,
        ny: usize,
    ) -> Result<Self, io::Error> {
        // Ensure parent directory exists
        let path = Path::new(&output_filepath);
        if let Some(parent_dir) = path.parent() {
            fs::create_dir_all(parent_dir)?; // Create parent dirs if they don't exist
            info!("Ensured output directory exists: {}", parent_dir.display());
        }

        Ok(Self {
            output_filepath,
            output_frequency,
            collected_data: Vec::new(),
            nx,
            ny,
        })
    }

    /// Determines if data should be collected for the given step based on frequency.
    pub fn should_collect(&self, step: usize, total_steps: usize, is_final_step: bool) -> bool {
        if let Some(freq) = self.output_frequency {
            if freq == 0 {
                // Frequency 0 means only initial (step 0) and final steps
                step == 0 || is_final_step
            } else {
                // Collect at step 0, multiples of freq, and the final step
                step == 0 || step % freq == 0 || is_final_step
            }
        } else {
            // No frequency means collect every step (including step 0 and final)
             true // Or decide if None means disabled - let's assume collect all if None freq
            // A better approach might be to not create the manager if output is disabled.
        }
    }

    /// Interpolates fields, flattens them, and adds a TimestepData entry.
    /// Takes the *current* Grid2D state.
    pub fn collect_timestep(
        &mut self,
        step: usize,
        time: f64,
        grid: &Grid2D, // Pass current grid state
    ) -> Result<(), io::Error> {

        if grid.dimensions.0 != self.nx || grid.dimensions.1 != self.ny {
             return Err(io::Error::new(io::ErrorKind::InvalidInput,
                format!("Grid dimensions ({},{}) do not match JsonOutputManager dimensions ({},{})",
                        grid.dimensions.0, grid.dimensions.1, self.nx, self.ny)));
        }

        info!("Collecting data for step {} (time {:.4})...", step, time);

        // Interpolate U and V to cell centers (Nx x Ny)
        let u_centers_mat = interpolate_u_to_cell_centers(&grid.u);
        let v_centers_mat = interpolate_v_to_cell_centers(&grid.v);
        let p_centers_mat = &grid.pressure; // Pressure is already Nx x Ny

        // Flatten matrices into Vec<f64> (column-major order)
        let u_centers_vec = u_centers_mat.as_slice().to_vec();
        let v_centers_vec = v_centers_mat.as_slice().to_vec();
        let p_centers_vec = p_centers_mat.as_slice().to_vec();

        // Create TimestepData struct
        let data_point = TimestepData {
            step,
            time,
            u_centers: u_centers_vec,
            v_centers: v_centers_vec,
            p_centers: p_centers_vec,
        };

        // Add to the collection
        self.collected_data.push(data_point);
        Ok(())
    }

    /// Writes the collected data and metadata to the final JSON file.
    pub fn write_final_output(
        &self,
        re: f64, // Pass final simulation parameters needed for metadata
        dt: f64,
        steps_completed: usize,
    ) -> Result<(), io::Error> {
        if self.collected_data.is_empty() {
            info!("No data collected, skipping JSON output to {}.", self.output_filepath);
            return Ok(());
        }

        info!("Writing collected data to JSON file: {}...", self.output_filepath);
        let output_start = std::time::Instant::now();

        // Create Metadata
        let metadata = Metadata {
            nx: self.nx,
            ny: self.ny,
            dx: 1.0 / self.nx as f64, // Assuming unit domain, calculate dx/dy here
            dy: 1.0 / self.ny as f64, // Or pass dx/dy if not unit domain
            re,
            dt,
            num_steps_completed: steps_completed,
            output_frequency: self.output_frequency,
        };

        // Create the final SimulationOutput struct (borrowing collected data)
        let output_data = SimulationOutput {
            metadata,
            data: &self.collected_data,
        };

        // Serialize to JSON string (pretty printed)
        let json_string = match serde_json::to_string_pretty(&output_data) {
            Ok(s) => s,
            Err(e) => {
                return Err(io::Error::new(io::ErrorKind::Other,
                    format!("Failed to serialize data to JSON: {}", e)));
            }
        };

        // Write to file
        let file = File::create(&self.output_filepath)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(json_string.as_bytes())?;
        writer.flush()?; // Ensure buffer is written

        info!("JSON output finished in {:.2}ms", output_start.elapsed().as_millis());
        Ok(())
    }
}


// --- Tests for json_io ---
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
        // Fill with some simple values
        grid.u.fill(1.0);
        grid.v.fill(2.0);
        grid.pressure.fill(3.0);
        grid
    }

    #[test]
    fn test_manager_new_creates_dir() -> io::Result<()> {
        let dir = tempdir()?;
        let filepath = dir.path().join("subdir").join("output.json");
        let filepath_str = filepath.to_str().unwrap().to_string();

        assert!(!dir.path().join("subdir").exists()); // Ensure subdir doesn't exist yet

        let _manager = JsonOutputManager::new(filepath_str, Some(10), 5, 5)?;

        assert!(dir.path().join("subdir").exists()); // Subdir should now exist

        dir.close()?;
        Ok(())
    }

     #[test]
    fn test_should_collect_logic() {
        // Freq = 10
        let manager_freq10 = JsonOutputManager{
            output_filepath: "".to_string(), output_frequency: Some(10), collected_data: vec![], nx: 10, ny: 10,
        };
        assert!(manager_freq10.should_collect(0, 100, false)); // Step 0
        assert!(!manager_freq10.should_collect(1, 100, false));
        assert!(!manager_freq10.should_collect(9, 100, false));
        assert!(manager_freq10.should_collect(10, 100, false)); // Multiple of freq
        assert!(!manager_freq10.should_collect(11, 100, false));
        assert!(manager_freq10.should_collect(20, 100, false));
        assert!(manager_freq10.should_collect(99, 100, true)); // Final step (even if not multiple)
        assert!(manager_freq10.should_collect(100, 100, true)); // Final step (and multiple)

         // Freq = 0 (Initial/Final only)
         let manager_freq0 = JsonOutputManager{
            output_filepath: "".to_string(), output_frequency: Some(0), collected_data: vec![], nx: 10, ny: 10,
         };
         assert!(manager_freq0.should_collect(0, 100, false)); // Step 0
         assert!(!manager_freq0.should_collect(1, 100, false));
         assert!(!manager_freq0.should_collect(50, 100, false));
         assert!(manager_freq0.should_collect(100, 100, true)); // Final step

        // Freq = None (Every step)
        let manager_freq_none = JsonOutputManager{
            output_filepath: "".to_string(), output_frequency: None, collected_data: vec![], nx: 10, ny: 10,
        };
        assert!(manager_freq_none.should_collect(0, 100, false));
        assert!(manager_freq_none.should_collect(1, 100, false));
        assert!(manager_freq_none.should_collect(50, 100, false));
        assert!(manager_freq_none.should_collect(100, 100, true));
    }

    #[test]
    fn test_collect_and_write() -> io::Result<()> {
        let nx = 3;
        let ny = 2;
        let dir = tempdir()?;
        let filepath = dir.path().join("collect_test.json");
        let filepath_str = filepath.to_str().unwrap().to_string();
        let mut manager = JsonOutputManager::new(filepath_str.clone(), None, nx, ny)?; // Collect every step

        let grid1 = create_dummy_grid(nx, ny);
        manager.collect_timestep(0, 0.0, &grid1)?;

        let mut grid2 = create_dummy_grid(nx, ny);
        grid2.pressure.fill(5.0); // Change something for step 1
        manager.collect_timestep(1, 0.1, &grid2)?;

        assert_eq!(manager.collected_data.len(), 2);
        assert_eq!(manager.collected_data[0].step, 0);
        assert_eq!(manager.collected_data[1].step, 1);
        assert_eq!(manager.collected_data[0].u_centers.len(), nx * ny);
        assert_eq!(manager.collected_data[1].p_centers.len(), nx * ny);

        // Write output
        manager.write_final_output(100.0, 0.1, 1)?; // Sim params, steps completed

        // Read back and verify basics
        let content = fs::read_to_string(filepath)?;
        let output: serde_json::Value = serde_json::from_str(&content)?;

        assert_eq!(output["metadata"]["nx"], nx);
        assert_eq!(output["metadata"]["ny"], ny);
        assert_eq!(output["metadata"]["re"], 100.0);
        assert_eq!(output["metadata"]["num_steps_completed"], 1);
        assert_eq!(output["data"].as_array().unwrap().len(), 2);
        // Check a value from the flattened data (remembering column-major)
        // p_centers for step 1 should be all 5.0
        let p1_flat = output["data"][1]["p_centers"].as_array().unwrap();
        assert_relative_eq!(p1_flat[0].as_f64().unwrap(), 5.0); // p[0,0]
        assert_relative_eq!(p1_flat[nx].as_f64().unwrap(), 5.0); // p[0,1]

        dir.close()?;
        Ok(())
    }
}