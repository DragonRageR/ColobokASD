use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::sync::{self, GpuFuture};
use vulkano::swapchain::{SwapchainPresentInfo, acquire_next_image};
use super::context::VulkanContext;
use super::swapchain::VulkanSwapchain;
use crate::gui::Canvas;

/// Графічний рендерер Vulkan: виконує передачу та представлення кадрів на відеокарті
pub struct VulkanRenderer {
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    transfer_buffers: [Option<Subbuffer<[u32]>>; 2],
    current_buffer_idx: usize,
}

impl VulkanRenderer {
    pub fn new(context: &VulkanContext) -> Self {
        Self {
            previous_frame_end: Some(Box::new(sync::now(context.device.clone())) as Box<dyn GpuFuture>),
            transfer_buffers: [None, None],
            current_buffer_idx: 0,
        }
    }

    /// Відображення поточного кадру Canvas на екрані через чергу Vulkan.
    /// Повертає `true`, якщо потрібно перевизначити swapchain (наприклад, змінено розмір вікна).
    pub fn draw(
        &mut self,
        context: &VulkanContext,
        swapchain_data: &mut VulkanSwapchain,
        canvas: &Canvas,
    ) -> bool {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let (image_index, suboptimal, acquire_future) = match acquire_next_image(swapchain_data.swapchain.clone(), None) {
            Ok(r) => r,
            Err(vulkano::Validated::Error(vulkano::VulkanError::OutOfDate)) => {
                self.previous_frame_end = Some(Box::new(sync::now(context.device.clone())) as Box<_>);
                return true;
            }
            Err(e) => {
                log::warn!("Could not acquire next image: {:?}. Triggering swapchain recreation...", e);
                self.previous_frame_end = Some(Box::new(sync::now(context.device.clone())) as Box<_>);
                return true;
            }
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            context.command_buffer_allocator.clone(),
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let image = swapchain_data.images[image_index as usize].clone();
        let req_len = canvas.pixels.len() as u64;

        // Подвійна буферизація (Double Buffering) трансферних буферів для усунення конфліктів доступу CPU/GPU
        self.current_buffer_idx = (self.current_buffer_idx + 1) % 2;

        let needs_new_buf = match &self.transfer_buffers[self.current_buffer_idx] {
            Some(b) => b.len() != req_len || b.write().is_err(),
            None => true,
        };

        if needs_new_buf {
            let new_buf = Buffer::new_slice::<u32>(
                context.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                req_len,
            )
            .unwrap();
            log::debug!(
                "Allocated new Vulkan transfer staging buffer slot {} ({} bytes)",
                self.current_buffer_idx,
                req_len * 4
            );
            self.transfer_buffers[self.current_buffer_idx] = Some(new_buf);
        }

        let buffer = self.transfer_buffers[self.current_buffer_idx].as_ref().unwrap();
        buffer.write().unwrap().copy_from_slice(&canvas.pixels);

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer.clone(), image))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self.previous_frame_end.take().unwrap()
            .join(acquire_future)
            .then_execute(context.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                context.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain_data.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
                suboptimal
            }
            Err(vulkano::Validated::Error(vulkano::VulkanError::OutOfDate)) => {
                log::debug!("Vulkan swapchain out of date, triggering recreation...");
                self.previous_frame_end = Some(Box::new(sync::now(context.device.clone())) as Box<_>);
                true
            }
            Err(e) => {
                log::error!("Vulkan rendering error occurred: {:?}", e);
                self.previous_frame_end = Some(Box::new(sync::now(context.device.clone())) as Box<_>);
                false
            }
        }
    }

    /// Скидання синхронізації GpuFuture при перевизначенні swapchain
    pub fn reset_frame_end(&mut self, context: &VulkanContext) {
        self.previous_frame_end = Some(Box::new(sync::now(context.device.clone())) as Box<_>);
    }
}
