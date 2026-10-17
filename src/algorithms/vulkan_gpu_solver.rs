use std::sync::Arc;
use std::time::Instant;
use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags, Queue};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineShaderStageCreateInfo, PipelineLayout};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, BufferContents};
use vulkano::sync::GpuFuture;

use crate::geometry::{Point, Square};
use super::solver::{AlgorithmStats, SquareSolver};
use super::randomized_search::RandomizedSearchSolver;

/// Структура точки для завантаження у VRAM графічного процесора
#[derive(BufferContents, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct GpuPoint {
    pub x: f32,
    pub y: f32,
    pub id: u32,
    pub pad: u32,
}

/// Параметри та метадані просторової сітки для передачі в шейдер
#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct GpuUniforms {
    pub num_points: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    pub max_squares: u32,
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub cell_size: f32,
    pub tolerance: f32,
    pub batch_start: u32,
    pub batch_count: u32,
    pub max_dist_sq: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// Знайдений чотирикутник (індекси вершин на GPU)
#[derive(BufferContents, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct GpuSquareIndices {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub l: u32,
}

/// Вихідний код WGSL шейдера для виконання на Vulkan Compute Pipeline.
/// Оптимізований під архітектуру Ada Lovelace (NVIDIA RTX 4050 Laptop):
/// - Розмір робочої групи 256 потоків = 8 апаратних варпів по 32 потоки
/// - Прямий доступ до локальної відеопам'яті VRAM (6GB)
/// - Швидке відсікання ребер за межами Bounding Box за O(1)
/// - Локалізований пошук у сітці (1-4 сусідні комірки замість 9)
/// - Батчування ітерацій (batching) для усунення Windows TDR та плавного boost dGPU
/// - Обмеження кількості кандидатів на точку (max_per_point) для захисту від гігантських витрат RAM
const COMPUTE_SHADER_WGSL: &str = r#"
struct Point {
    x: f32,
    y: f32,
    id: u32,
    pad: u32,
};

struct Uniforms {
    num_points: u32,
    grid_w: u32,
    grid_h: u32,
    max_squares: u32,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    cell_size: f32,
    tolerance: f32,
    batch_start: u32,
    batch_count: u32,
    max_dist_sq: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> u: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<Point>;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> cell_points: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_count: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> out_squares: array<vec4<u32>>;

// Пошук найближчої точки до цільових координат (tx, ty) у комірці сітки (cx, cy)
fn find_point_in_cell(tx: f32, ty: f32, cx: i32, cy: i32, min_idx: u32, exclude_j: u32, tol_sq: f32) -> u32 {
    if (cx < 0 || cy < 0 || cx >= i32(u.grid_w) || cy >= i32(u.grid_h)) {
        return 0xFFFFFFFFu;
    }
    let cell_idx = u32(cy) * u.grid_w + u32(cx);
    let start = cell_offsets[cell_idx];
    let end = cell_offsets[cell_idx + 1u];

    var best_idx = 0xFFFFFFFFu;
    var best_dist = tol_sq;

    for (var idx = start; idx < end; idx = idx + 1u) {
        let pt_idx = cell_points[idx];
        if (pt_idx > min_idx && pt_idx != exclude_j) {
            let p = points[pt_idx];
            let ddx = p.x - tx;
            let ddy = p.y - ty;
            let d2 = ddx * ddx + ddy * ddy;
            if (d2 <= best_dist) {
                best_dist = d2;
                best_idx = pt_idx;
            }
        }
    }
    return best_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_i = global_id.x;
    if (local_i >= u.batch_count) {
        return;
    }
    let i = u.batch_start + local_i;
    if (i >= u.num_points) {
        return;
    }

    let p_a = points[i];
    let tol = u.tolerance;
    let tol_sq = tol * tol;

    for (var j = i + 1u; j < u.num_points; j = j + 1u) {
        let p_b = points[j];
        let dx = p_b.x - p_a.x;
        let dy = p_b.y - p_a.y;
        let dist_sq = dx * dx + dy * dy;

        // Ігноруємо надто близькі або просторово віддалені точки
        if (dist_sq < 1.0 || dist_sq > u.max_dist_sq) {
            continue;
        }

        // Поворот ребра AB на +90 градусів: (-dy, dx)
        let c1_x = p_b.x - dy;
        let c1_y = p_b.y + dx;

        // Швидка O(1) перевірка виходу вершини C1 за межі простору
        if (c1_x < u.min_x - tol || c1_x > u.max_x + tol ||
            c1_y < u.min_y - tol || c1_y > u.max_y + tol) {
            continue;
        }

        let d1_x = p_a.x - dy;
        let d1_y = p_a.y + dx;

        // Швидка O(1) перевірка виходу вершини D1 за межі простору
        if (d1_x < u.min_x - tol || d1_x > u.max_x + tol ||
            d1_y < u.min_y - tol || d1_y > u.max_y + tol) {
            continue;
        }

        let min_cx = i32(floor((c1_x - tol - u.min_x) / u.cell_size));
        let max_cx = i32(floor((c1_x + tol - u.min_x) / u.cell_size));
        let min_cy = i32(floor((c1_y - tol - u.min_y) / u.cell_size));
        let max_cy = i32(floor((c1_y + tol - u.min_y) / u.cell_size));

        // Пошук точки C у релевантних комірках сітки (1-4 комірки замість 9)
        var k_found = 0xFFFFFFFFu;
        for (var cy = min_cy; cy <= max_cy; cy = cy + 1) {
            for (var cx = min_cx; cx <= max_cx; cx = cx + 1) {
                if (k_found == 0xFFFFFFFFu) {
                    k_found = find_point_in_cell(c1_x, c1_y, cx, cy, i, j, tol_sq);
                }
            }
        }

        // Якщо точка C знайдена, перевіряємо точку D
        if (k_found != 0xFFFFFFFFu) {
            let min_dx = i32(floor((d1_x - tol - u.min_x) / u.cell_size));
            let max_dx = i32(floor((d1_x + tol - u.min_x) / u.cell_size));
            let min_dy = i32(floor((d1_y - tol - u.min_y) / u.cell_size));
            let max_dy = i32(floor((d1_y + tol - u.min_y) / u.cell_size));

            var l_found = 0xFFFFFFFFu;
            for (var cy = min_dy; cy <= max_dy; cy = cy + 1) {
                for (var cx = min_dx; cx <= max_dx; cx = cx + 1) {
                    if (l_found == 0xFFFFFFFFu) {
                        l_found = find_point_in_cell(d1_x, d1_y, cx, cy, i, j, tol_sq);
                    }
                }
            }

            // Якщо всі 4 вершини квадрата знайдені, додаємо квадрат через атомарний лічильник
            if (l_found != 0xFFFFFFFFu && l_found != k_found) {
                let slot = atomicAdd(&out_count, 1u);
                if (slot < u.max_squares) {
                    out_squares[slot] = vec4<u32>(i, j, k_found, l_found);
                }
            }
        }
    }
}
"#;

/// Контекст дискретної відеокарти dGPU для обчислень Vulkan Compute
pub struct VulkanComputeDevice {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub pipeline: Arc<ComputePipeline>,
    pub gpu_name: String,
    pub is_discrete: bool,
    pub vram_mb: f64,
}

impl VulkanComputeDevice {
    /// Ініціалізація та налаштування Vulkan Compute Pipeline з пріоритетом Discrete GPU (NVIDIA RTX 4050)
    pub fn init_dgpu() -> Result<Self, String> {
        let library = VulkanLibrary::new().map_err(|e| format!("Failed to load Vulkan library: {e}"))?;
        let instance = Instance::new(library, InstanceCreateInfo::default())
            .map_err(|e| format!("Failed to create Vulkan instance: {e}"))?;

        // Шукаємо підходящий фізичний пристрій з підтримкою черги COMPUTE, віддаючи найвищий пріоритет DiscreteGpu
        let mut physical_devices: Vec<(Arc<PhysicalDevice>, u32, bool)> = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("Failed to enumerate physical devices: {e}"))?
            .filter_map(|p| {
                let is_discrete = p.properties().device_type == PhysicalDeviceType::DiscreteGpu;
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
                    .map(|q_idx| (p, q_idx as u32, is_discrete))
            })
            .collect();

        physical_devices.sort_by_key(|(_, _, is_discrete)| if *is_discrete { 0 } else { 1 });

        let (physical_device, queue_family_index, is_discrete) = physical_devices
            .into_iter()
            .next()
            .ok_or_else(|| "No Vulkan-compatible GPU with compute queue support found".to_string())?;

        let gpu_name = physical_device.properties().device_name.clone();

        let mut vram_bytes = 0u64;
        for heap in &physical_device.memory_properties().memory_heaps {
            if heap.flags.intersects(vulkano::memory::MemoryHeapFlags::DEVICE_LOCAL) {
                vram_bytes = vram_bytes.max(heap.size);
            }
        }
        let vram_mb = vram_bytes as f64 / (1024.0 * 1024.0);

        log::info!(
            "Vulkan dGPU Compute initialized on '{}' (Type: {:?}, Dedicated VRAM: {:.1} MB, Warp/Subgroup size: 32)",
            gpu_name,
            physical_device.properties().device_type,
            vram_mb
        );

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| format!("Failed to create Vulkan compute device: {e}"))?;

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Компіляція WGSL шейдера через чистий Naga у SPIR-V (версія 1.3 для повної сумісності з Vulkan)
        let module = naga::front::wgsl::parse_str(COMPUTE_SHADER_WGSL)
            .map_err(|e| format!("WGSL parse error: {e}"))?;
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .map_err(|e| format!("WGSL validation error: {e}"))?;

        let mut spv = Vec::new();
        let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        })
        .map_err(|e| format!("SPIR-V writer init error: {e}"))?;

        writer
            .write(&module, &info, None, &None, &mut spv)
            .map_err(|e| format!("SPIR-V write error: {e}"))?;

        let shader_module = unsafe {
            vulkano::shader::ShaderModule::new(
                device.clone(),
                vulkano::shader::ShaderModuleCreateInfo::new(&spv),
            )
            .map_err(|e| format!("Failed to create Vulkan shader module: {e}"))?
        };

        let entry_point = shader_module
            .entry_point("main")
            .ok_or_else(|| "Shader entry point 'main' not found".to_string())?;

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .map_err(|e| format!("Pipeline layout create info error: {e}"))?,
        )
        .map_err(|e| format!("Failed to create pipeline layout: {e}"))?;

        let pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .map_err(|e| format!("Failed to create compute pipeline: {e}"))?;

        Ok(Self {
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            pipeline,
            gpu_name,
            is_discrete,
            vram_mb,
        })
    }
}

/// Апаратний розв'язувач задачі пошуку квадратів на dGPU (NVIDIA RTX 4050) через Vulkan Compute
pub struct VulkanGpuSolver {
    pub hash_variants: usize,
}

impl VulkanGpuSolver {
    pub fn new(hash_variants: usize) -> Self {
        Self { hash_variants }
    }

    /// Знаходження кандидатів у квадрати за допомогою Vulkan Compute Pipeline на dGPU
    pub fn find_candidates_gpu(
        gpu: &VulkanComputeDevice,
        points: &[Point],
        tolerance: f64,
    ) -> Result<Vec<Square>, String> {
        let n = points.len();
        if n < 4 {
            return Ok(Vec::new());
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for p in points {
            let px = p.x as f32;
            let py = p.y as f32;
            if px < min_x { min_x = px; }
            if px > max_x { max_x = px; }
            if py < min_y { min_y = py; }
            if py > max_y { max_y = py; }
        }

        let span = ((max_x - min_x).max(max_y - min_y)).max(10.0);
        let tol_f32 = tolerance as f32;
        let cell_size = (span / (n as f32).sqrt()).max(tol_f32 * 2.0).max(2.0);

        let grid_w = (((max_x - min_x) / cell_size).floor() as u32) + 1;
        let grid_h = (((max_y - min_y) / cell_size).floor() as u32) + 1;
        let total_cells = (grid_w * grid_h) as usize;

        // Будуємо 1D сітку на CPU для відправки у VRAM (O(N) побудова)
        let mut cell_counts = vec![0u32; total_cells];
        let mut pt_cells = Vec::with_capacity(n);

        for p in points {
            let cx = (((p.x as f32 - min_x) / cell_size).floor() as u32).min(grid_w - 1);
            let cy = (((p.y as f32 - min_y) / cell_size).floor() as u32).min(grid_h - 1);
            let cid = cy * grid_w + cx;
            pt_cells.push(cid);
            cell_counts[cid as usize] += 1;
        }

        let mut cell_offsets = vec![0u32; total_cells + 1];
        for c in 0..total_cells {
            cell_offsets[c + 1] = cell_offsets[c] + cell_counts[c];
        }

        let mut insert_pos = cell_offsets.clone();
        let mut cell_points = vec![0u32; n];
        for (pt_idx, &cid) in pt_cells.iter().enumerate() {
            let pos = insert_pos[cid as usize] as usize;
            cell_points[pos] = pt_idx as u32;
            insert_pos[cid as usize] += 1;
        }

        let gpu_points: Vec<GpuPoint> = points
            .iter()
            .enumerate()
            .map(|(idx, p)| GpuPoint {
                x: p.x as f32,
                y: p.y as f32,
                id: (idx + 1) as u32,
                pad: 0,
            })
            .collect();

        let max_dist = if n <= 2_000 {
            1e12f32
        } else {
            280.0f32
        };
        let max_dist_sq = max_dist * max_dist;
        let max_squares = 3_000_000u32;

        let buf_points = Buffer::from_iter(
            gpu.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            gpu_points,
        )
        .map_err(|e| format!("Failed to allocate points buffer: {e}"))?;

        let buf_offsets = Buffer::from_iter(
            gpu.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            cell_offsets,
        )
        .map_err(|e| format!("Failed to allocate cell offsets buffer: {e}"))?;

        let buf_cell_pts = Buffer::from_iter(
            gpu.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            cell_points,
        )
        .map_err(|e| format!("Failed to allocate cell points buffer: {e}"))?;

        let buf_count = Buffer::from_data(
            gpu.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            0u32,
        )
        .map_err(|e| format!("Failed to allocate counter buffer: {e}"))?;

        let buf_squares = Buffer::new_slice::<GpuSquareIndices>(
            gpu.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            max_squares as vulkano::DeviceSize,
        )
        .map_err(|e| format!("Failed to allocate output squares buffer: {e}"))?;

        let pipeline_layout = gpu.pipeline.layout();
        let set_layout = pipeline_layout.set_layouts().first().unwrap();

        // Батчування ітерацій (4 000 точок на батч) для усунення Windows TDR та плавного розгону частот dGPU
        let batch_size = 4_000u32;
        let num_batches = ((n as u32) + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_count = (n as u32 - batch_start).min(batch_size);

            log::debug!("Starting GPU batch {}/{} (start={}, count={})...", batch_idx + 1, num_batches, batch_start, batch_count);
            let t_batch = std::time::Instant::now();

            let uniforms = GpuUniforms {
                num_points: n as u32,
                grid_w,
                grid_h,
                max_squares,
                min_x,
                min_y,
                max_x,
                max_y,
                cell_size,
                tolerance: tol_f32,
                batch_start,
                batch_count,
                max_dist_sq,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };

            let buf_uniforms = Buffer::from_data(
                gpu.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                uniforms,
            )
            .map_err(|e| format!("Failed to allocate uniforms buffer: {e}"))?;

            let descriptor_set = DescriptorSet::new(
                gpu.descriptor_set_allocator.clone(),
                set_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, buf_uniforms),
                    WriteDescriptorSet::buffer(1, buf_points.clone()),
                    WriteDescriptorSet::buffer(2, buf_offsets.clone()),
                    WriteDescriptorSet::buffer(3, buf_cell_pts.clone()),
                    WriteDescriptorSet::buffer(4, buf_count.clone()),
                    WriteDescriptorSet::buffer(5, buf_squares.clone()),
                ],
                [],
            )
            .map_err(|e| format!("Failed to create descriptor set: {e}"))?;

            let mut builder = AutoCommandBufferBuilder::primary(
                gpu.command_buffer_allocator.clone(),
                gpu.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .map_err(|e| format!("Failed to create command buffer builder: {e}"))?;

            let workgroups = (batch_count + 255) / 256;

            unsafe {
                builder
                    .bind_pipeline_compute(gpu.pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        vulkano::pipeline::PipelineBindPoint::Compute,
                        pipeline_layout.clone(),
                        0,
                        descriptor_set,
                    )
                    .unwrap()
                    .dispatch([workgroups, 1, 1])
                    .unwrap();
            }

            let command_buffer = builder
                .build()
                .map_err(|e| format!("Failed to build command buffer: {e}"))?;

            let future = vulkano::sync::now(gpu.device.clone())
                .then_execute(gpu.queue.clone(), command_buffer)
                .map_err(|e| format!("Failed to execute command buffer: {e}"))?
                .then_signal_fence_and_flush()
                .map_err(|e| format!("Failed to flush fence: {e}"))?;

            future.wait(None).map_err(|e| format!("GPU batch wait failed: {e:?}"))?;
            log::debug!("GPU batch {}/{} finished in {:.2} ms", batch_idx + 1, num_batches, t_batch.elapsed().as_secs_f64() * 1000.0);
        }

        let total_detected = *buf_count.read().map_err(|e| format!("Read count failed: {e}"))?;
        if total_detected > max_squares {
            log::warn!(
                "GPU candidates exceeded buffer limit: detected {} > max_squares {}. Buffer capacity reached!",
                total_detected, max_squares
            );
        }
        let found_count = total_detected.min(max_squares);
        let squares_read = buf_squares.read().map_err(|e| format!("Read squares failed: {e}"))?;

        use rayon::prelude::*;

        let slice = &squares_read[..found_count as usize];
        let mut raw_squares: Vec<[u32; 4]> = slice
            .par_iter()
            .map(|sq| {
                let mut arr = [sq.i, sq.j, sq.k, sq.l];
                arr.sort_unstable();
                arr
            })
            .collect();

        raw_squares.par_sort_unstable();
        raw_squares.dedup();

        // Обмежуємо кількість кандидатів до 2 000 000 для радикальної економії RAM (< 400 MB замість 10 GB)
        if raw_squares.len() > 2_000_000 {
            raw_squares.truncate(2_000_000);
        }

        let candidates: Vec<Square> = raw_squares
            .into_par_iter()
            .filter_map(|[i, j, k, l]| {
                let ui = i as usize;
                let uj = j as usize;
                let uk = k as usize;
                let ul = l as usize;
                if ui < n && uj < n && uk < n && ul < n {
                    Some(Square::new(
                        [ui, uj, uk, ul],
                        [points[ui], points[uj], points[uk], points[ul]],
                    ))
                } else {
                    None
                }
            })
            .collect();

        Ok(candidates)
    }
}

impl SquareSolver for VulkanGpuSolver {
    fn name(&self) -> &'static str {
        "Vulkan dGPU Compute (NVIDIA RTX 4050 - 2560 Cores / Warps)"
    }

    fn description(&self) -> &'static str {
        "Апаратне прискорення детекції квадратів через Vulkan Compute Pipeline на 2560 CUDA-ядрах dGPU NVIDIA GeForce RTX 4050 з паралельними варпами по 32 потоки та стохастичним мультистартом."
    }

    fn complexity(&self) -> &'static str {
        "O(n² / (SM·Warp)) час GPU, O(n + M) VRAM"
    }

    fn solve(&self, points: &[Point], tolerance: f64) -> (Vec<Square>, AlgorithmStats) {
        log::info!(
            "Starting Vulkan dGPU solver on {} points (tolerance: {:.2}, iterations: {})...",
            points.len(),
            tolerance,
            self.hash_variants
        );

        let start_total = Instant::now();

        // 1. Ініціалізація dGPU пристрою
        let gpu_init_start = Instant::now();
        let gpu_device = match VulkanComputeDevice::init_dgpu() {
            Ok(dev) => dev,
            Err(e) => {
                log::error!("dGPU Vulkan initialization failed: {}. Falling back to CPU Rayon solver...", e);
                let fallback = RandomizedSearchSolver::new(self.hash_variants);
                return fallback.solve(points, tolerance);
            }
        };

        log::info!(
            "Selected GPU for compute: '{}' (Discrete: {}, VRAM: {:.1} MB, init: {:.2} ms)",
            gpu_device.gpu_name,
            gpu_device.is_discrete,
            gpu_device.vram_mb,
            gpu_init_start.elapsed().as_secs_f64() * 1000.0
        );

        // 2. Детекція кандидатів на Vulkan Compute Pipeline dGPU
        let start_detect = Instant::now();
        let candidates = match Self::find_candidates_gpu(&gpu_device, points, tolerance) {
            Ok(c) => c,
            Err(e) => {
                log::error!("GPU candidate detection failed: {}. Falling back to CPU...", e);
                super::candidate_detector::find_candidate_squares(points, tolerance)
            }
        };
        let detection_time_ms = start_detect.elapsed().as_secs_f64() * 1000.0;

        log::info!(
            "dGPU candidate detection finished in {:.2} ms (found {} candidates)",
            detection_time_ms,
            candidates.len()
        );

        // 3. Стохастичний мультистарт максимальної незалежної підмножини (Rayon CPU)
        let start_select = Instant::now();
        let chosen = RandomizedSearchSolver::select_best_randomized(
            &candidates,
            points.len(),
            self.hash_variants,
        );
        let selection_time_ms = start_select.elapsed().as_secs_f64() * 1000.0;

        let total_time_ms = start_total.elapsed().as_secs_f64() * 1000.0;

        let memory_kb = ((points.len() * std::mem::size_of::<GpuPoint>()
            + candidates.len() * std::mem::size_of::<GpuSquareIndices>()) as f64)
            / 1024.0;

        let stats = AlgorithmStats {
            algorithm_name: format!("Vulkan dGPU ({})", gpu_device.gpu_name),
            complexity: self.complexity().to_string(),
            detection_time_ms,
            selection_time_ms,
            total_time_ms,
            points_count: points.len(),
            candidates_found: candidates.len(),
            disjoint_squares_count: chosen.len(),
            memory_kb,
            hash_variants: self.hash_variants,
        };

        log::info!(
            "Vulkan dGPU solver completed in {:.2} ms (Detection: {:.2} ms, Selection: {:.2} ms, Found: {} independent squares)",
            total_time_ms,
            detection_time_ms,
            selection_time_ms,
            chosen.len()
        );

        (chosen, stats)
    }
}
