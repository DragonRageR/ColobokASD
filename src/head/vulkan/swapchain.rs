use std::sync::Arc;
use vulkano::image::{ImageUsage, Image};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use winit::window::Window;
use super::context::VulkanContext;

/// Обгортка ланцюжка обміну кадрів Vulkan (Swapchain)
pub struct VulkanSwapchain {
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<Image>>,
}

impl VulkanSwapchain {
    /// Створення нового ланцюжка обміну з підтримкою Mailbox / Immediate для максимальної частоти кадрів (FPS)
    pub fn new(context: &VulkanContext, window: &Window) -> Self {
        let surface_capabilities = context.device
            .physical_device()
            .surface_capabilities(&context.surface, Default::default())
            .unwrap();
        
        let surface_format = context.device
            .physical_device()
            .surface_formats(&context.surface, Default::default())
            .unwrap()[0];

        // Обираємо найшвидший доступний режим синхронізації без вертикальної затримки (V-Sync drop)
        let present_mode = context.device
            .physical_device()
            .surface_present_modes(&context.surface, Default::default())
            .map(|modes| {
                if modes.contains(&vulkano::swapchain::PresentMode::Mailbox) {
                    vulkano::swapchain::PresentMode::Mailbox
                } else if modes.contains(&vulkano::swapchain::PresentMode::Immediate) {
                    vulkano::swapchain::PresentMode::Immediate
                } else {
                    vulkano::swapchain::PresentMode::Fifo
                }
            })
            .unwrap_or(vulkano::swapchain::PresentMode::Fifo);

        log::info!("Vulkan swapchain present mode selected: {:?}", present_mode);

        let (swapchain, images) = Swapchain::new(
            context.device.clone(),
            context.surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format: surface_format.0,
                image_color_space: surface_format.1,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                present_mode,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap();

        log::info!(
            "Vulkan swapchain created successfully (image count: {}, format: {:?})",
            images.len(),
            surface_format.0
        );

        Self { swapchain, images }
    }

    /// Перестворення ланцюжка обміну при зміні розмірів вікна
    pub fn recreate(&mut self, window: &Window) -> Result<(), ()> {
        let new_dimensions = window.inner_size();
        if new_dimensions.width == 0 || new_dimensions.height == 0 {
            return Err(());
        }

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: new_dimensions.into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(_) => return Err(()),
        };

        log::debug!(
            "Vulkan swapchain recreated successfully with extent: [{}, {}]",
            new_dimensions.width,
            new_dimensions.height
        );

        self.swapchain = new_swapchain;
        self.images = new_images;
        Ok(())
    }
}
