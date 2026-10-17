use std::sync::Arc;
use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::gui::{Canvas, UiState};
use crate::gui::ui_state::{next_zoom_in, next_zoom_out};
use crate::head::vulkan::{VulkanContext, VulkanRenderer, VulkanSwapchain};

/// Головна структура програми, що обробляє події вікна, стан інтерфейсу та Vulkan-конвеєр
pub struct App {
    instance: Arc<Instance>,
    window: Option<Arc<Window>>,

    // Графічні компоненти Vulkan
    vk_context: Option<VulkanContext>,
    vk_swapchain: Option<VulkanSwapchain>,
    vk_renderer: Option<VulkanRenderer>,

    pub window_resized: bool,
    recreate_swapchain: bool,
    pub needs_redraw: bool,

    // Стан інтерфейсу та 2D полотно
    pub canvas: Canvas,
    pub ui_state: UiState,
}

impl App {
    pub fn with_instance(instance: Arc<Instance>) -> Self {
        Self {
            instance,
            window: None,
            vk_context: None,
            vk_swapchain: None,
            vk_renderer: None,
            window_resized: false,
            recreate_swapchain: false,
            needs_redraw: true,
            canvas: Canvas::new(1200, 750),
            ui_state: UiState::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window_attrs = Window::default_attributes()
                .with_title("АСД: Курсова робота (Завдання №23 - Незалежні Квадрати)")
                .with_inner_size(LogicalSize::new(1200.0, 750.0));

            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

            let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();

            let context = VulkanContext::new(self.instance.clone(), surface);
            self.ui_state.gpu_name = context.device_name.clone();
            self.ui_state.is_igpu = context.is_igpu;

            let swapchain = VulkanSwapchain::new(&context, &window);
            let renderer = VulkanRenderer::new(&context);

            let inner = window.inner_size();
            self.canvas.resize(inner.width as usize, inner.height as usize);

            window.request_redraw();
            self.window = Some(window);
            self.vk_context = Some(context);
            self.vk_swapchain = Some(swapchain);
            self.vk_renderer = Some(renderer);

            log::info!("Vulkan graphics context and swapchain initialized successfully.");
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Window close requested. Terminating application event loop...");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    log::debug!("Window resized to {}x{}", size.width, size.height);
                    self.canvas.resize(size.width as usize, size.height as usize);
                    self.window_resized = true;
                    self.needs_redraw = true;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let mx = position.x as i32;
                let my = position.y as i32;
                self.ui_state.mouse_screen_x = mx;
                self.ui_state.mouse_screen_y = my;

                if self.ui_state.is_dragging_view {
                    let dx = mx - self.ui_state.drag_start_x;
                    let dy = my - self.ui_state.drag_start_y;
                    self.ui_state.pan_x += dx as f64;
                    self.ui_state.pan_y += dy as f64;
                    self.ui_state.drag_start_x = mx;
                    self.ui_state.drag_start_y = my;
                    self.needs_redraw = true;
                } else {
                    // Перевіряємо чи курсор над правою панеллю для оновлення підсвічування кнопок (hover)
                    let width = self.canvas.width as i32;
                    let sidebar_x = self.ui_state.sidebar_x(width);
                    if mx >= sidebar_x {
                        self.needs_redraw = true;
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let width = self.canvas.width as i32;
                let sidebar_x = self.ui_state.sidebar_x(width);
                let x = self.ui_state.mouse_screen_x;
                let y = self.ui_state.mouse_screen_y;
                self.needs_redraw = true;

                match (state, button) {
                    (ElementState::Pressed, MouseButton::Left) => {
                        if x >= sidebar_x {
                            self.ui_state.handle_click(x, y, false, width);
                        } else {
                            // На полотні клік скидає фокус введення та вмикає панорамування (Pan)
                            self.ui_state.is_input_focused = false;
                            self.ui_state.is_hash_focused = false;
                            self.ui_state.is_dragging_view = true;
                            self.ui_state.drag_start_x = x;
                            self.ui_state.drag_start_y = y;
                        }
                    }
                    (ElementState::Pressed, MouseButton::Right) => {
                        if x < sidebar_x {
                            // Права кнопка миші на полотні також забезпечує панорамування (Pan)
                            self.ui_state.is_input_focused = false;
                            self.ui_state.is_hash_focused = false;
                            self.ui_state.is_dragging_view = true;
                            self.ui_state.drag_start_x = x;
                            self.ui_state.drag_start_y = y;
                        }
                    }
                    (ElementState::Released, MouseButton::Left) | (ElementState::Released, MouseButton::Right) => {
                        self.ui_state.is_dragging_view = false;
                    }
                    _ => (),
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let width = self.canvas.width as i32;
                let sidebar_x = self.ui_state.sidebar_x(width);
                let mx = self.ui_state.mouse_screen_x;

                // Масштабування дискретними кратними кроками, коли курсор над полотном
                if mx < sidebar_x {
                    let scroll_amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y as f64,
                        MouseScrollDelta::PixelDelta(pos) => (pos.y / 50.0) as f64,
                    };

                    let prev_zoom = self.ui_state.zoom;
                    let new_zoom = if scroll_amount > 0.0 {
                        next_zoom_in(prev_zoom)
                    } else if scroll_amount < 0.0 {
                        next_zoom_out(prev_zoom)
                    } else {
                        prev_zoom
                    };

                    if (new_zoom - prev_zoom).abs() > 1e-5 {
                        let cur_x = mx as f64;
                        let cur_y = self.ui_state.mouse_screen_y as f64;

                        // Зберігаємо світові координати точки під курсором миші незмінними під час зміни зуму
                        self.ui_state.pan_x = cur_x - (cur_x - self.ui_state.pan_x) * (new_zoom / prev_zoom);
                        self.ui_state.pan_y = cur_y - (cur_y - self.ui_state.pan_y) * (new_zoom / prev_zoom);
                        self.ui_state.zoom = new_zoom;
                        log::debug!("Canvas zoom level changed: {:.2}x -> {:.2}x", prev_zoom, new_zoom);
                        self.needs_redraw = true;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match &event.logical_key {
                        Key::Named(NamedKey::Backspace) => {
                            if self.ui_state.handle_backspace() {
                                self.needs_redraw = true;
                            }
                        }
                        Key::Named(NamedKey::Delete) => {
                            if self.ui_state.handle_delete() {
                                self.needs_redraw = true;
                            }
                        }
                        Key::Named(NamedKey::ArrowLeft) => {
                            if self.ui_state.move_cursor_left() {
                                self.needs_redraw = true;
                            }
                        }
                        Key::Named(NamedKey::ArrowRight) => {
                            if self.ui_state.move_cursor_right() {
                                self.needs_redraw = true;
                            }
                        }
                        Key::Named(NamedKey::Home) => {
                            if self.ui_state.move_cursor_home() {
                                self.needs_redraw = true;
                            }
                        }
                        Key::Named(NamedKey::End) => {
                            if self.ui_state.move_cursor_end() {
                                self.needs_redraw = true;
                            }
                        }
                        Key::Named(NamedKey::Enter) => {
                            if self.ui_state.handle_enter() {
                                self.needs_redraw = true;
                            }
                        }
                        Key::Named(NamedKey::Escape) => {
                            if self.ui_state.is_input_focused || self.ui_state.is_hash_focused {
                                self.ui_state.is_input_focused = false;
                                self.ui_state.is_hash_focused = false;
                                self.needs_redraw = true;
                            }
                        }
                        Key::Character(s) => {
                            let mut changed = false;
                            for c in s.chars() {
                                if self.ui_state.handle_char(c) {
                                    changed = true;
                                }
                            }
                            if changed {
                                self.needs_redraw = true;
                            }
                        }
                        _ => (),
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.draw();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let finished = self.ui_state.poll_background_computation();
        if finished {
            self.needs_redraw = true;
        }

        // Під час обчислень на CPU тримаємо таймер 16 мс (60 FPS) для постійного прокачування повідомлень ОС
        if self.ui_state.is_computing {
            self.ui_state.anim_tick = self.ui_state.anim_tick.wrapping_add(1);
            if self.ui_state.anim_tick % 2 == 0 {
                self.needs_redraw = true;
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::wait_duration(
                std::time::Duration::from_millis(16),
            ));
        } else if self.ui_state.is_input_focused || self.ui_state.is_hash_focused {
            self.ui_state.anim_tick = self.ui_state.anim_tick.wrapping_add(1);
            if self.ui_state.anim_tick % 20 == 0 {
                self.needs_redraw = true;
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::wait_duration(
                std::time::Duration::from_millis(50),
            ));
        } else {
            // Енергоефективне очікування подій ОС при спокої
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
        }

        if self.needs_redraw {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

impl App {
    fn draw(&mut self) {
        self.needs_redraw = false;

        let window = self.window.as_ref().unwrap();
        let context = self.vk_context.as_ref().unwrap();
        let swapchain = self.vk_swapchain.as_mut().unwrap();
        let renderer = self.vk_renderer.as_mut().unwrap();

        if self.window_resized || self.recreate_swapchain {
            self.recreate_swapchain = false;

            match swapchain.recreate(window) {
                Ok(_) => {
                    self.window_resized = false;
                    renderer.reset_frame_end(context);
                }
                Err(_) => return, // Вікно згорнуто (розмір 0x0)
            }
        }

        // 1. Оновлюємо стан полотна Canvas
        self.ui_state.render(&mut self.canvas);

        // 2. Відправляємо підготовлений буфер на відмальовування через Vulkan
        let needs_recreate = renderer.draw(context, swapchain, &self.canvas);
        if needs_recreate {
            self.recreate_swapchain = true;
        }
    }
}

pub fn run_app() {
    let event_loop = EventLoop::new().unwrap();
    let library = VulkanLibrary::new().expect("Не знайдено бібліотеку Vulkan (vulkan-1.dll)");

    // Отримуємо розширення поверхні Vulkan для вікна
    let required_extensions = Surface::required_extensions(&event_loop).unwrap();

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("Не вдалося створити екземпляр Vulkan");

    log::info!(
        "Vulkan library loaded successfully. Vulkan API version: {}",
        instance.api_version()
    );
    log::info!("--------------------------------------------------");

    let mut app = App::with_instance(instance);

    log::info!("Launching application window event loop...");
    event_loop.run_app(&mut app).unwrap();
}
