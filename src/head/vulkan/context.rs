use std::sync::Arc;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags, DeviceExtensions, Queue};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::instance::Instance;
use vulkano::swapchain::Surface;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::memory::allocator::StandardMemoryAllocator;

/// Контекст Vulkan: керує логічним пристроєм, чергою команд та розподільниками пам'яті
pub struct VulkanContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub device_name: String,
    pub is_igpu: bool,
}

impl VulkanContext {
    /// Ініціалізація графічного контексту Vulkan з пріоритетом інтегрованого відеоадаптера (iGPU)
    pub fn new(instance: Arc<Instance>, surface: Arc<Surface>) -> Self {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::IntegratedGpu => 0, // Найвищий пріоритет: iGPU для візуалізації
                PhysicalDeviceType::DiscreteGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("Не знайдено підходящої відеокарти з підтримкою Vulkan");

        log::info!(
            "Selected Vulkan graphics adapter: '{}' (Type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type
        );
        log::info!(
            "Vulkan graphics queue initialized on queue family index: {}",
            queue_family_index
        );

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let device_name = physical_device.properties().device_name.clone();
        let is_igpu = physical_device.properties().device_type == PhysicalDeviceType::IntegratedGpu;

        Self {
            device,
            queue,
            surface,
            command_buffer_allocator,
            memory_allocator,
            device_name,
            is_igpu,
        }
    }
}
