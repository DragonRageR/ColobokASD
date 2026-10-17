pub mod candidate_detector;
pub mod randomized_search;
pub mod solver;
pub mod vulkan_gpu_solver;

pub use candidate_detector::find_candidate_squares;
pub use randomized_search::RandomizedSearchSolver;
pub use solver::{AlgorithmStats, SquareSolver};
pub use vulkan_gpu_solver::VulkanGpuSolver;

