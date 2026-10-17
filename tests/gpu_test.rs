use std::sync::Arc;
use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::pipeline::Pipeline;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryTypeFilter};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::sync::GpuFuture;

#[test]
fn test_detect_nvidia_gpu() {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(library, InstanceCreateInfo::default()).unwrap();
    
    let mut found_discrete = false;
    for p in instance.enumerate_physical_devices().unwrap() {
        println!("Found device: {} ({:?})", p.properties().device_name, p.properties().device_type);
        if p.properties().device_type == PhysicalDeviceType::DiscreteGpu {
            found_discrete = true;
            println!("  Max storage buffer range: {:.2} MB", p.properties().max_storage_buffer_range as f64 / (1024.0 * 1024.0));
            println!("  Max compute work group invocations: {}", p.properties().max_compute_work_group_invocations);
            println!("  Memory heaps:");
            for heap in &p.memory_properties().memory_heaps {
                println!("    Heap size: {:.2} MB (device local: {})", 
                    heap.size as f64 / (1024.0 * 1024.0),
                    heap.flags.intersects(vulkano::memory::MemoryHeapFlags::DEVICE_LOCAL)
                );
            }
        }
    }
    assert!(found_discrete, "Discrete GPU must be present");
}

#[test]
fn test_naga_wgsl_to_spirv() {
    let wgsl = r#"
    @group(0) @binding(0) var<storage, read_write> data: array<u32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        data[idx] = data[idx] * 2u;
    }
    "#;

    let module = naga::front::wgsl::parse_str(wgsl).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).unwrap();
    let mut spv = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    }).unwrap();
    writer.write(&module, &info, None, &None, &mut spv).unwrap();
    assert!(!spv.is_empty());
    println!("Compiled SPIR-V words: {}", spv.len());
}

#[test]
fn test_vulkan_compute_execution() {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(library, InstanceCreateInfo::default()).unwrap();

    // Вибираємо саме dGPU (NVIDIA RTX 4050)
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter_map(|p| {
            let is_discrete = p.properties().device_type == PhysicalDeviceType::DiscreteGpu;
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|q_idx| (p, q_idx as u32, is_discrete))
        })
        .max_by_key(|(_, _, is_discrete)| *is_discrete)
        .map(|(p, q, _)| (p, q))
        .expect("No GPU with compute support found");

    println!("Selected compute device: {}", physical_device.properties().device_name);

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    ).unwrap();

    let queue = queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

    let wgsl = r#"
    @group(0) @binding(0) var<storage, read_write> data: array<u32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        if (idx < arrayLength(&data)) {
            data[idx] = data[idx] * 3u + 1u;
        }
    }
    "#;

    let module = naga::front::wgsl::parse_str(wgsl).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).unwrap();
    let mut spv = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    }).unwrap();
    writer.write(&module, &info, None, &None, &mut spv).unwrap();

    let shader_module = unsafe {
        vulkano::shader::ShaderModule::new(
            device.clone(),
            vulkano::shader::ShaderModuleCreateInfo::new(&spv),
        ).unwrap()
    };

    let entry_point = shader_module.entry_point("main").unwrap();
    let stage = vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point);
    let layout = vulkano::pipeline::PipelineLayout::new(
        device.clone(),
        vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    ).unwrap();

    let pipeline = vulkano::pipeline::compute::ComputePipeline::new(
        device.clone(),
        None,
        vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(stage, layout),
    ).unwrap();

    // Створюємо буфер даних
    let data_slice: Vec<u32> = (0..256).map(|i| i as u32).collect();
    let buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_slice,
    ).unwrap();

    let pipeline_layout = pipeline.layout();
    let set_layout = pipeline_layout.set_layouts().first().unwrap();
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        set_layout.clone(),
        [WriteDescriptorSet::buffer(0, buffer.clone())],
        [],
    ).unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    unsafe {
        builder
            .bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .dispatch([4, 1, 1])
            .unwrap();
    }

    let command_buffer = builder.build().unwrap();

    let future = vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let read_guard = buffer.read().unwrap();
    for (i, &val) in read_guard.iter().enumerate() {
        assert_eq!(val, (i as u32) * 3 + 1);
    }
    println!("Successfully computed on GPU! Sample result: data[10] = {}", read_guard[10]);
}

#[test]
fn test_candidate_detection_shader_compiles() {
    let wgsl = r#"
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
        cell_size: f32,
        tolerance: f32,
    };

    @group(0) @binding(0) var<storage, read> u: Uniforms;
    @group(0) @binding(1) var<storage, read> points: array<Point>;
    @group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
    @group(0) @binding(3) var<storage, read> cell_points: array<u32>;
    @group(0) @binding(4) var<storage, read_write> out_count: atomic<u32>;
    @group(0) @binding(5) var<storage, read_write> out_squares: array<vec4<u32>>;

    fn find_point_in_cell(tx: f32, ty: f32, cx: i32, cy: i32, min_idx: u32, exclude_j: u32, tol_sq: f32) -> u32 {
        if (cx < 0 || cy < 0 || cx >= i32(u.grid_w) || cy >= i32(u.grid_h)) {
            return 0xFFFFFFFFu;
        }
        let cell_idx = u32(cy) * u.grid_w + u32(cx);
        let start = cell_offsets[cell_idx];
        let end = cell_offsets[cell_idx + 1u];

        var best_idx = 0xFFFFFFFFu;
        var best_dist = tol_sq;

        for (var i = start; i < end; i = i + 1u) {
            let pt_idx = cell_points[i];
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
        let i = global_id.x;
        if (i >= u.num_points) {
            return;
        }

        let p_a = points[i];
        let tol_sq = u.tolerance * u.tolerance;

        for (var j = i + 1u; j < u.num_points; j = j + 1u) {
            let p_b = points[j];
            let dx = p_b.x - p_a.x;
            let dy = p_b.y - p_a.y;
            let dist_sq = dx * dx + dy * dy;

            if (dist_sq < 1.0) {
                continue;
            }

            let c1_x = p_b.x - dy;
            let c1_y = p_b.y + dx;

            let cx = i32(floor((c1_x - u.min_x) / u.cell_size));
            let cy = i32(floor((c1_y - u.min_y) / u.cell_size));

            var k_found = 0xFFFFFFFFu;
            for (var s_dy = -1; s_dy <= 1; s_dy = s_dy + 1) {
                for (var s_dx = -1; s_dx <= 1; s_dx = s_dx + 1) {
                    if (k_found == 0xFFFFFFFFu) {
                        k_found = find_point_in_cell(c1_x, c1_y, cx + s_dx, cy + s_dy, i, j, tol_sq);
                    }
                }
            }

            if (k_found != 0xFFFFFFFFu) {
                let d1_x = p_a.x - dy;
                let d1_y = p_a.y + dx;

                let dx_cell = i32(floor((d1_x - u.min_x) / u.cell_size));
                let dy_cell = i32(floor((d1_y - u.min_y) / u.cell_size));

                var l_found = 0xFFFFFFFFu;
                for (var s_dy = -1; s_dy <= 1; s_dy = s_dy + 1) {
                    for (var s_dx = -1; s_dx <= 1; s_dx = s_dx + 1) {
                        if (l_found == 0xFFFFFFFFu) {
                            l_found = find_point_in_cell(d1_x, d1_y, dx_cell + s_dx, dy_cell + s_dy, i, j, tol_sq);
                        }
                    }
                }

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

    let module = naga::front::wgsl::parse_str(wgsl).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).unwrap();
    let mut spv = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    }).unwrap();
    writer.write(&module, &info, None, &None, &mut spv).unwrap();
    assert!(!spv.is_empty());
    println!("Candidate square detector SPIR-V compiled successfully ({} words)!", spv.len());
}

#[derive(vulkano::buffer::BufferContents, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct GpuPoint {
    pub x: f32,
    pub y: f32,
    pub id: u32,
    pub pad: u32,
}

#[derive(vulkano::buffer::BufferContents, Clone, Copy, Debug)]
#[repr(C)]
struct GpuUniforms {
    pub num_points: u32,
    pub grid_w: u32,
    pub grid_h: u32,
    pub max_squares: u32,
    pub min_x: f32,
    pub min_y: f32,
    pub cell_size: f32,
    pub tolerance: f32,
}

#[derive(vulkano::buffer::BufferContents, Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct GpuSquareIndices {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub l: u32,
}

#[test]
fn test_gpu_candidate_detector_full() {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(library, InstanceCreateInfo::default()).unwrap();

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter_map(|p| {
            let is_discrete = p.properties().device_type == PhysicalDeviceType::DiscreteGpu;
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|q_idx| (p, q_idx as u32, is_discrete))
        })
        .max_by_key(|(_, _, is_discrete)| *is_discrete)
        .map(|(p, q, _)| (p, q))
        .expect("No GPU with compute support found");

    println!("Executing candidate detector on: {}", physical_device.properties().device_name);

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    ).unwrap();

    let queue = queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));

    // Тестові точки: квадрат (0, 1, 2, 3) + повернений квадрат (4, 5, 6, 7) + шум (8, 9)
    let raw_pts = vec![
        // Квадрат 1
        (10.0, 10.0), // 0: A
        (20.0, 10.0), // 1: B
        (20.0, 20.0), // 2: C
        (10.0, 20.0), // 3: D
        // Квадрат 2 (повернутий на 45 градусів)
        (50.0, 50.0), // 4
        (60.0, 60.0), // 5
        (50.0, 70.0), // 6
        (40.0, 60.0), // 7
        // Шум
        (100.0, 100.0), // 8
        (120.0, 140.0), // 9
    ];

    let n = raw_pts.len();
    let tolerance = 0.5f32;

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for &(x, y) in &raw_pts {
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }

    let span = ((max_x - min_x).max(max_y - min_y)).max(10.0);
    let cell_size = (span / (n as f32).sqrt()).max(tolerance * 2.0).max(2.0);

    let grid_w = (((max_x - min_x) / cell_size).floor() as u32) + 1;
    let grid_h = (((max_y - min_y) / cell_size).floor() as u32) + 1;
    let total_cells = (grid_w * grid_h) as usize;

    let mut cell_counts = vec![0u32; total_cells];
    let mut pt_cells = Vec::with_capacity(n);

    for &(x, y) in &raw_pts {
        let cx = (((x - min_x) / cell_size).floor() as u32).min(grid_w - 1);
        let cy = (((y - min_y) / cell_size).floor() as u32).min(grid_h - 1);
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

    let gpu_points: Vec<GpuPoint> = raw_pts
        .iter()
        .enumerate()
        .map(|(id, &(x, y))| GpuPoint {
            x,
            y,
            id: (id + 1) as u32,
            pad: 0,
        })
        .collect();

    let uniforms = GpuUniforms {
        num_points: n as u32,
        grid_w,
        grid_h,
        max_squares: 1000,
        min_x,
        min_y,
        cell_size,
        tolerance,
    };

    let wgsl = r#"
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
        cell_size: f32,
        tolerance: f32,
    };

    @group(0) @binding(0) var<storage, read> u: Uniforms;
    @group(0) @binding(1) var<storage, read> points: array<Point>;
    @group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
    @group(0) @binding(3) var<storage, read> cell_points: array<u32>;
    @group(0) @binding(4) var<storage, read_write> out_count: atomic<u32>;
    @group(0) @binding(5) var<storage, read_write> out_squares: array<vec4<u32>>;

    fn find_point_in_cell(tx: f32, ty: f32, cx: i32, cy: i32, min_idx: u32, exclude_j: u32, tol_sq: f32) -> u32 {
        if (cx < 0 || cy < 0 || cx >= i32(u.grid_w) || cy >= i32(u.grid_h)) {
            return 0xFFFFFFFFu;
        }
        let cell_idx = u32(cy) * u.grid_w + u32(cx);
        let start = cell_offsets[cell_idx];
        let end = cell_offsets[cell_idx + 1u];

        var best_idx = 0xFFFFFFFFu;
        var best_dist = tol_sq;

        for (var i = start; i < end; i = i + 1u) {
            let pt_idx = cell_points[i];
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
        let i = global_id.x;
        if (i >= u.num_points) {
            return;
        }

        let p_a = points[i];
        let tol_sq = u.tolerance * u.tolerance;

        for (var j = i + 1u; j < u.num_points; j = j + 1u) {
            let p_b = points[j];
            let dx = p_b.x - p_a.x;
            let dy = p_b.y - p_a.y;
            let dist_sq = dx * dx + dy * dy;

            if (dist_sq < 1.0) {
                continue;
            }

            let c1_x = p_b.x - dy;
            let c1_y = p_b.y + dx;

            let cx = i32(floor((c1_x - u.min_x) / u.cell_size));
            let cy = i32(floor((c1_y - u.min_y) / u.cell_size));

            var k_found = 0xFFFFFFFFu;
            for (var s_dy = -1; s_dy <= 1; s_dy = s_dy + 1) {
                for (var s_dx = -1; s_dx <= 1; s_dx = s_dx + 1) {
                    if (k_found == 0xFFFFFFFFu) {
                        k_found = find_point_in_cell(c1_x, c1_y, cx + s_dx, cy + s_dy, i, j, tol_sq);
                    }
                }
            }

            if (k_found != 0xFFFFFFFFu) {
                let d1_x = p_a.x - dy;
                let d1_y = p_a.y + dx;

                let dx_cell = i32(floor((d1_x - u.min_x) / u.cell_size));
                let dy_cell = i32(floor((d1_y - u.min_y) / u.cell_size));

                var l_found = 0xFFFFFFFFu;
                for (var s_dy = -1; s_dy <= 1; s_dy = s_dy + 1) {
                    for (var s_dx = -1; s_dx <= 1; s_dx = s_dx + 1) {
                        if (l_found == 0xFFFFFFFFu) {
                            l_found = find_point_in_cell(d1_x, d1_y, dx_cell + s_dx, dy_cell + s_dy, i, j, tol_sq);
                        }
                    }
                }

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

    let module = naga::front::wgsl::parse_str(wgsl).unwrap();
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator.validate(&module).unwrap();
    let mut spv = Vec::new();
    let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    }).unwrap();
    writer.write(&module, &info, None, &None, &mut spv).unwrap();

    let shader_module = unsafe {
        vulkano::shader::ShaderModule::new(
            device.clone(),
            vulkano::shader::ShaderModuleCreateInfo::new(&spv),
        ).unwrap()
    };

    let entry_point = shader_module.entry_point("main").unwrap();
    let stage = vulkano::pipeline::PipelineShaderStageCreateInfo::new(entry_point);
    let layout = vulkano::pipeline::PipelineLayout::new(
        device.clone(),
        vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    ).unwrap();

    let pipeline = vulkano::pipeline::compute::ComputePipeline::new(
        device.clone(),
        None,
        vulkano::pipeline::compute::ComputePipelineCreateInfo::stage_layout(stage, layout),
    ).unwrap();

    // Створюємо буфери
    let buf_uniforms = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
        uniforms,
    ).unwrap();

    let buf_points = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
        gpu_points,
    ).unwrap();

    let buf_offsets = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
        cell_offsets,
    ).unwrap();

    let buf_cell_pts = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default() },
        cell_points,
    ).unwrap();

    let buf_count = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS, ..Default::default() },
        0u32,
    ).unwrap();

    let empty_squares = vec![GpuSquareIndices { i: 0, j: 0, k: 0, l: 0 }; 1000];
    let buf_squares = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo { usage: BufferUsage::STORAGE_BUFFER, ..Default::default() },
        AllocationCreateInfo { memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS, ..Default::default() },
        empty_squares,
    ).unwrap();

    let pipeline_layout = pipeline.layout();
    let set_layout = pipeline_layout.set_layouts().first().unwrap();
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        set_layout.clone(),
        [
            WriteDescriptorSet::buffer(0, buf_uniforms),
            WriteDescriptorSet::buffer(1, buf_points),
            WriteDescriptorSet::buffer(2, buf_offsets),
            WriteDescriptorSet::buffer(3, buf_cell_pts),
            WriteDescriptorSet::buffer(4, buf_count.clone()),
            WriteDescriptorSet::buffer(5, buf_squares.clone()),
        ],
        [],
    ).unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    let workgroups = ((n as u32) + 255) / 256;

    unsafe {
        builder
            .bind_pipeline_compute(pipeline.clone())
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

    let command_buffer = builder.build().unwrap();

    let future = vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let found_count = *buf_count.read().unwrap();
    println!("GPU Candidate detector found {} squares on NVIDIA dGPU!", found_count);
    assert_eq!(found_count, 2, "Must find exactly 2 squares (one regular, one rotated)");

    let squares_read = buf_squares.read().unwrap();
    for idx in 0..found_count as usize {
        let sq = &squares_read[idx];
        println!("  Square {}: i={}, j={}, k={}, l={}", idx + 1, sq.i, sq.j, sq.k, sq.l);
    }
}

#[test]
fn test_vulkan_gpu_solver_api() {
    use ASDKyrsova::algorithms::{SquareSolver, VulkanGpuSolver};
    use ASDKyrsova::io::FileManager;

    let points = FileManager::generate_sample_dataset(100, 500.0, 500.0, 5);
    let solver = VulkanGpuSolver::new(20);
    let (squares, stats) = solver.solve(&points, 0.5);

    println!("Vulkan dGPU solver test results:");
    println!("  Algorithm: {}", stats.algorithm_name);
    println!("  Total time: {:.2} ms (Detection: {:.2} ms, Selection: {:.2} ms)", 
        stats.total_time_ms, stats.detection_time_ms, stats.selection_time_ms);
    println!("  Candidates found: {}", stats.candidates_found);
    println!("  Disjoint squares: {}", stats.disjoint_squares_count);

    assert!(stats.disjoint_squares_count >= 1, "Must find at least 1 square");
    assert_eq!(squares.len(), stats.disjoint_squares_count);
}

#[test]
fn test_vulkan_gpu_solver_large() {
    use ASDKyrsova::algorithms::{SquareSolver, VulkanGpuSolver};
    use ASDKyrsova::io::FileManager;

    let points = FileManager::generate_sample_dataset(10_000, 1000.0, 800.0, 2_500);
    let solver = VulkanGpuSolver::new(5);
    let (squares, stats) = solver.solve(&points, 0.5);
    println!("Large dataset test finished: found {} disjoint squares, candidates: {}", squares.len(), stats.candidates_found);
    assert!(stats.disjoint_squares_count >= 1);
    assert_eq!(squares.len(), stats.disjoint_squares_count);
}

#[test]
fn test_vulkan_gpu_solver_50k() {
    use ASDKyrsova::algorithms::{SquareSolver, VulkanGpuSolver};
    use ASDKyrsova::io::FileManager;

    let points = FileManager::generate_sample_dataset(50_000, 1000.0, 800.0, 12_500);
    let solver = VulkanGpuSolver::new(40);
    let (squares, stats) = solver.solve(&points, 0.5);

    // Перевіримо скільки з 12500 вбудованих квадратів було виявлено
    let dev = ASDKyrsova::algorithms::vulkan_gpu_solver::VulkanComputeDevice::init_dgpu().unwrap();
    let candidates = VulkanGpuSolver::find_candidates_gpu(&dev, &points, 0.5).unwrap();

    let mut detected_injected = 0;
    use std::collections::HashSet;
    let mut cand_set = HashSet::new();
    for sq in &candidates {
        let mut idxs = sq.indices;
        idxs.sort_unstable();
        cand_set.insert(idxs);
    }

    for k in 0..12_500 {
        let base = k * 4;
        let mut expected = [base, base + 1, base + 2, base + 3];
        expected.sort_unstable();
        if cand_set.contains(&expected) {
            detected_injected += 1;
        }
    }

    println!("=== 50k Points Test Results ===");
    println!("  Total points: {}", stats.points_count);
    println!("  Injected squares detected: {} / 12500 ({:.2}%)", 
        detected_injected, (detected_injected as f64 / 12500.0) * 100.0);
    println!("  Candidates found: {}", stats.candidates_found);
    println!("  Disjoint figures: {} ({:.2}% error)", 
        squares.len(), ((12500.0 - squares.len() as f64) / 12500.0) * 100.0);

    // Аналіз степенів конфліктності (base_degrees)
    let mut point_to_squares: Vec<Vec<u32>> = vec![Vec::new(); points.len()];
    for (sq_idx, sq) in candidates.iter().enumerate() {
        for &pt in &sq.indices {
            if pt < points.len() {
                point_to_squares[pt].push(sq_idx as u32);
            }
        }
    }

    let mut inj_degs = Vec::new();
    let mut acc_degs = Vec::new();

    for sq in &candidates {
        let mut idxs = sq.indices;
        idxs.sort_unstable();
        let mut deg: u32 = 0;
        for &pt in &sq.indices {
            if pt < points.len() {
                deg += (point_to_squares[pt].len() as u32).saturating_sub(1);
            }
        }
        if idxs[0] % 4 == 0 && idxs[1] == idxs[0] + 1 && idxs[2] == idxs[0] + 2 && idxs[3] == idxs[0] + 3 {
            inj_degs.push(deg);
        } else {
            acc_degs.push(deg);
        }
    }

    let avg_inj_deg = inj_degs.iter().map(|&d| d as f64).sum::<f64>() / inj_degs.len().max(1) as f64;
    let avg_acc_deg = acc_degs.iter().map(|&d| d as f64).sum::<f64>() / acc_degs.len().max(1) as f64;
    println!("  Avg Injected Deg: {:.2} (min: {:?}, max: {:?})", 
        avg_inj_deg, inj_degs.iter().min(), inj_degs.iter().max());
    println!("  Avg Accidental Deg: {:.2} (min: {:?}, max: {:?})", 
        avg_acc_deg, acc_degs.iter().min(), acc_degs.iter().max());

    let mut chosen_injected = 0;
    let mut chosen_accidental = 0;
    let mut sample_acc = Vec::new();
    for sq in &squares {
        let mut idxs = sq.indices;
        idxs.sort_unstable();
        if idxs[0] % 4 == 0 && idxs[1] == idxs[0] + 1 && idxs[2] == idxs[0] + 2 && idxs[3] == idxs[0] + 3 {
            chosen_injected += 1;
        } else {
            chosen_accidental += 1;
            if sample_acc.len() < 5 {
                sample_acc.push(sq.clone());
            }
        }
    }
    println!("  Chosen: {} injected, {} accidental", chosen_injected, chosen_accidental);
    for (i, sq) in sample_acc.iter().enumerate() {
        println!("    Sample Accidental {}: side={:.1}, indices={:?}, pts={:?}", 
            i+1, sq.side_length(), sq.indices, sq.points.iter().map(|p| (p.x, p.y)).collect::<Vec<_>>());
    }
    println!("  Detection time: {:.2} ms", stats.detection_time_ms);
    println!("  Selection time: {:.2} ms", stats.selection_time_ms);
    println!("  Total time: {:.2} ms", stats.total_time_ms);
    println!("  Memory estimation: {:.2} MB", stats.memory_kb / 1024.0);
}





