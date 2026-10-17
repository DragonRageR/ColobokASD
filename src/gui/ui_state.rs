use std::sync::mpsc::{channel, Receiver};
use std::thread;
use log::{debug, error, info, warn};

use crate::algorithms::{AlgorithmStats, RandomizedSearchSolver, SquareSolver, VulkanGpuSolver};
use crate::geometry::{Point, Square};
use crate::gui::canvas::{Canvas, Color};
use crate::io::FileManager;

/// Ширина правої панелі керування та статистики (у пікселях)
pub const SIDEBAR_WIDTH: i32 = 380;

/// Допустимі фіксовані рівні масштабування полотна (кратні 0.25 або цілі числа).
pub const ZOOM_LEVELS: &[f64] = &[
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
];

/// Вибір активного обчислювального алгоритму
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmChoice {
    /// Vulkan Compute Pipeline на дискретній відеокарті (NVIDIA RTX 4050 з 2560 ядрами та варпами)
    VulkanDgpu,
    /// Багатопотоковий CPU стохастичний мультистарт на пулі Rayon
    CpuRayon,
}

/// Перехід до наступного більшого рівня зуму
pub fn next_zoom_in(current: f64) -> f64 {
    for &lvl in ZOOM_LEVELS {
        if lvl > current + 0.01 {
            return lvl;
        }
    }
    *ZOOM_LEVELS.last().unwrap()
}

/// Перехід до попереднього меншого рівня зуму
pub fn next_zoom_out(current: f64) -> f64 {
    for &lvl in ZOOM_LEVELS.iter().rev() {
        if lvl < current - 0.01 {
            return lvl;
        }
    }
    *ZOOM_LEVELS.first().unwrap()
}

/// Форматування значення зуму кратним числом без небажаних дробових залишків (наприклад: "1x", "1.5x", "0.75x")
pub fn format_zoom(zoom: f64) -> String {
    if (zoom - zoom.round()).abs() < 1e-4 {
        format!("{:.0}x", zoom)
    } else if (zoom * 10.0 - (zoom * 10.0).round()).abs() < 1e-4 {
        format!("{:.1}x", zoom)
    } else {
        format!("{:.2}x", zoom)
    }
}

/// Ідентифікатори інтерактивних дій інтерфейсу
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionId {
    /// Запуск основного алгоритму пошуку квадратів
    RunAlg,
    /// Вибір алгоритму Vulkan dGPU (NVIDIA RTX 4050)
    SetAlgorithmGpu,
    /// Вибір алгоритму CPU Rayon (12 потоків)
    SetAlgorithmCpu,
    /// Вибір та завантаження файлу через нативний діалог ОС
    PickAndLoadFile,
    /// Збереження результатів через нативний діалог ОС
    PickAndSaveFile,
    /// Центрування та скидання виду полотна
    ResetView,
    /// Очищення полотна від точок та фігур
    ClearPoints,
    /// Генерація N точок з поля вводу
    GenerateCustom,
    /// Застосування довільно введеної кількості хеш варіантів
    ApplyHashVariants,
    /// Зменшення кількості хеш варіантів на 10 (мін. 1)
    DecHashVariants,
    /// Збільшення кількості хеш варіантів на 10
    IncHashVariants,
    /// Встановлення фіксованого значення хеш варіантів
    SetHashVariants(usize),
}

/// Опис візуальної кнопки графічного інтерфейсу
pub struct UiButton {
    pub id: ActionId,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub label: &'static str,
    pub bg_color: Color,
}

/// Результат фонового виконання обчислень
pub struct ComputationResult {
    pub squares: Vec<Square>,
    pub stats: AlgorithmStats,
}

/// Повний стан графічного інтерфейсу, полотна та взаємодії користувача
pub struct UiState {
    // Дані задачі
    pub points: Vec<Point>,
    pub candidate_squares: Vec<Square>,
    pub disjoint_squares: Vec<Square>,
    pub stats: AlgorithmStats,

    // Фонове виконання алгоритму (окремий потік для збереження чутливості вікна)
    pub is_computing: bool,
    computation_rx: Option<Receiver<ComputationResult>>,
    pub anim_tick: u64,

    // Перегляд та масштабування
    pub pan_x: f64,
    pub pan_y: f64,
    pub zoom: f64,

    // Інтерактивність та курсор миші
    pub mouse_screen_x: i32,
    pub mouse_screen_y: i32,
    pub is_dragging_view: bool,
    pub drag_start_x: i32,
    pub drag_start_y: i32,
    pub hovered_point_idx: Option<usize>,

    // Повідомлення статусу та налаштування
    pub status_message: String,
    pub tolerance: f64,
    pub hash_variants: usize,
    pub square_points_mask: Vec<bool>,

    // Введення кількості точок для генерації (ліміт знято)
    pub point_input_str: String,
    pub input_cursor_pos: usize,
    pub is_input_focused: bool,

    // Довільне введення кількості хеш-варіантів
    pub hash_input_str: String,
    pub hash_cursor_pos: usize,
    pub is_hash_focused: bool,

    pub gpu_name: String,
    pub is_igpu: bool,
    pub algorithm_choice: AlgorithmChoice,
}

impl UiState {
    pub fn new() -> Self {
        let mut state = Self {
            points: Vec::new(),
            candidate_squares: Vec::new(),
            disjoint_squares: Vec::new(),
            stats: AlgorithmStats::default(),
            is_computing: false,
            computation_rx: None,
            anim_tick: 0,
            pan_x: 40.0,
            pan_y: 40.0,
            zoom: 1.0,
            mouse_screen_x: 0,
            mouse_screen_y: 0,
            is_dragging_view: false,
            drag_start_x: 0,
            drag_start_y: 0,
            hovered_point_idx: None,
            status_message: "Готово до роботи. Додайте точки або оберіть файл.".to_string(),
            tolerance: 0.5,
            hash_variants: 40,
            square_points_mask: Vec::new(),
            point_input_str: "500".to_string(),
            input_cursor_pos: 3,
            is_input_focused: false,
            hash_input_str: "40".to_string(),
            hash_cursor_pos: 2,
            is_hash_focused: false,
            gpu_name: "Intel(R) UHD Graphics (iGPU)".to_string(),
            is_igpu: true,
            algorithm_choice: AlgorithmChoice::VulkanDgpu,
        };

        // Завантажуємо data.txt або input.txt за замовчуванням
        if let Ok(pts) = FileManager::load_points("data.txt") {
            let count = pts.len();
            state.points = pts;
            state.reset_view();
            state.status_message = format!("Завантажено {} точок з 'data.txt'. Натисніть 'Запустити'.", count);
            info!("UiState initialized: loaded default points file 'data.txt' ({} points)", count);
        } else if let Ok(pts) = FileManager::load_points("input.txt") {
            let count = pts.len();
            state.points = pts;
            state.reset_view();
            state.status_message = format!("Завантажено {} точок з 'input.txt'. Натисніть 'Запустити'.", count);
            info!("UiState initialized: loaded default points file 'input.txt' ({} points)", count);
        } else {
            info!("UiState initialized: no default dataset found, generating initial sample dataset in memory");
            state.points = FileManager::generate_sample_dataset(24, 1000.0, 800.0, 6);
            state.reset_view();
            state.status_message = "Згенеровано 24 точки. Натисніть 'Запустити'.".to_string();
        }

        state
    }

    /// Обчислення координати початку правої панелі
    pub fn sidebar_x(&self, window_width: i32) -> i32 {
        (window_width - SIDEBAR_WIDTH).max(400)
    }

    /// Єдине джерело координат та параметрів усіх кнопок інтерфейсу
    pub fn get_buttons(&self, window_width: i32) -> Vec<UiButton> {
        let sx = self.sidebar_x(window_width);
        let pad_x = sx + 15;
        let w = window_width - sx - 30;
        let mut buttons = Vec::new();
        let half_w = (w - 10) / 2;

        // 0. Вибір алгоритму (dGPU NVIDIA RTX 4050 vs CPU Rayon 12-Threads)
        let is_gpu = self.algorithm_choice == AlgorithmChoice::VulkanDgpu;
        buttons.push(UiButton {
            id: ActionId::SetAlgorithmGpu,
            x: pad_x,
            y: 120,
            w: half_w,
            h: 26,
            label: if is_gpu { "* dGPU 4050 *" } else { "dGPU RTX 4050" },
            bg_color: if is_gpu { Color::rgb(35, 134, 54) } else { Color::rgb(48, 54, 61) },
        });

        buttons.push(UiButton {
            id: ActionId::SetAlgorithmCpu,
            x: pad_x + half_w + 10,
            y: 120,
            w: half_w,
            h: 26,
            label: if !is_gpu { "* CPU 12-T *" } else { "CPU (Rayon 12T)" },
            bg_color: if !is_gpu { Color::rgb(31, 111, 235) } else { Color::rgb(48, 54, 61) },
        });

        // 1. Довільне введення кількості хеш варіантів
        let hash_input_w = 110.min(w / 2);
        let hash_btn_x = pad_x + hash_input_w + 8;
        let hash_btn_w = w - hash_input_w - 8;
        buttons.push(UiButton {
            id: ActionId::ApplyHashVariants,
            x: hash_btn_x,
            y: 178,
            w: hash_btn_w,
            h: 28,
            label: "Встановити",
            bg_color: Color::rgb(31, 111, 235),
        });

        // 2. Основні дії
        buttons.push(UiButton {
            id: ActionId::RunAlg,
            x: pad_x,
            y: 248,
            w: half_w,
            h: 30,
            label: if self.is_computing { "Обчислення..." } else { "Запустити" },
            bg_color: if self.is_computing { Color::rgb(150, 100, 30) } else { Color::rgb(35, 134, 54) },
        });

        buttons.push(UiButton {
            id: ActionId::ResetView,
            x: pad_x + half_w + 10,
            y: 248,
            w: half_w,
            h: 30,
            label: "Центрувати",
            bg_color: Color::rgb(70, 80, 95),
        });

        // 3. Робота з файлами через нативні системні діалоги ОС
        buttons.push(UiButton {
            id: ActionId::PickAndLoadFile,
            x: pad_x,
            y: 284,
            w: half_w,
            h: 30,
            label: "Обрати файл...",
            bg_color: Color::rgb(31, 111, 235),
        });

        buttons.push(UiButton {
            id: ActionId::PickAndSaveFile,
            x: pad_x + half_w + 10,
            y: 284,
            w: half_w,
            h: 30,
            label: "Зберегти файл...",
            bg_color: Color::rgb(137, 87, 229),
        });

        // 4. Очищення полотна
        buttons.push(UiButton {
            id: ActionId::ClearPoints,
            x: pad_x,
            y: 320,
            w,
            h: 26,
            label: "Очистити всi точки",
            bg_color: Color::rgb(218, 54, 51),
        });

        // 5. Введення та генерація точок
        let input_w = 120.min(w / 2);
        let btn_x = pad_x + input_w + 10;
        let btn_w = w - input_w - 10;

        buttons.push(UiButton {
            id: ActionId::GenerateCustom,
            x: btn_x,
            y: 376,
            w: btn_w,
            h: 28,
            label: "Згенерувати",
            bg_color: Color::rgb(35, 134, 54),
        });

        buttons
    }

    /// Отримання координат прямокутника поля вводу кількості точок
    pub fn get_input_rect(&self, window_width: i32) -> (i32, i32, i32, i32) {
        let sx = self.sidebar_x(window_width);
        let pad_x = sx + 15;
        let w = window_width - sx - 30;
        let input_w = 120.min(w / 2);
        (pad_x, 376, input_w, 28)
    }

    /// Отримання координат прямокутника поля довільного вводу кількості хеш-варіантів
    pub fn get_hash_input_rect(&self, window_width: i32) -> (i32, i32, i32, i32) {
        let sx = self.sidebar_x(window_width);
        let pad_x = sx + 15;
        let w = window_width - sx - 30;
        let input_w = 110.min(w / 2);
        (pad_x, 178, input_w, 28)
    }

    /// Перевірка результатів фонового потоку (non-blocking)
    pub fn poll_background_computation(&mut self) -> bool {
        if let Some(rx) = &self.computation_rx {
            if let Ok(res) = rx.try_recv() {
                self.disjoint_squares = res.squares;
                self.stats = res.stats;
                self.is_computing = false;
                self.computation_rx = None;

                self.square_points_mask = vec![false; self.points.len() + 10];
                for sq in &self.disjoint_squares {
                    for &pt_idx in &sq.indices {
                        if pt_idx < self.square_points_mask.len() {
                            self.square_points_mask[pt_idx] = true;
                        }
                    }
                }

                // Автоматично зберігаємо результат у вихідні файли відповідно до вимог
                let _ = FileManager::save_results("Squares.txt", &self.disjoint_squares, &self.stats);
                let _ = FileManager::save_results("output.txt", &self.disjoint_squares, &self.stats);
                let _ = FileManager::save_results("Rectangles.txt", &self.disjoint_squares, &self.stats);
                let _ = FileManager::save_results("Rombs.txt", &self.disjoint_squares, &self.stats);

                info!(
                    "Background worker finished: found {} disjoint squares in {:.2} ms (candidates: {}, hash variants: {})",
                    self.disjoint_squares.len(),
                    self.stats.total_time_ms,
                    self.stats.candidates_found,
                    self.stats.hash_variants
                );

                self.status_message = format!(
                    "Готово! Знайдено {} незалежних квадратів за {:.2} мс (збережено у Squares.txt)",
                    self.disjoint_squares.len(),
                    self.stats.total_time_ms
                );
                return true;
            }
        }
        false
    }

    /// Запуск алгоритму пошуку квадратів в окремому фоновому потоці
    pub fn run_selected_algorithm(&mut self) {
        if self.is_computing {
            warn!("Cannot start square search: computation is already running in background!");
            return;
        }

        if self.points.len() < 4 {
            warn!(
                "Cannot start square search: dataset has only {} points (minimum 4 required)",
                self.points.len()
            );
            self.disjoint_squares.clear();
            self.candidate_squares.clear();
            self.square_points_mask.clear();
            self.status_message = "Для побудови квадратів потрібно щонайменше 4 точки!".to_string();
            return;
        }

        self.is_computing = true;
        self.status_message = "Обчислення квадратів у фоновому потоці...".to_string();

        let (tx, rx) = channel();
        self.computation_rx = Some(rx);

        let points = self.points.clone();
        let tolerance = self.tolerance;
        let hash_variants = self.hash_variants;

        let algorithm_choice = self.algorithm_choice;

        info!(
            "Spawning background solver worker thread for {} points (algorithm: {:?}, tolerance: {:.2}, hash variants: {})...",
            points.len(),
            algorithm_choice,
            tolerance,
            hash_variants
        );

        thread::spawn(move || {
            let (squares, stats) = match algorithm_choice {
                AlgorithmChoice::VulkanDgpu => {
                    let solver = VulkanGpuSolver::new(hash_variants);
                    solver.solve(&points, tolerance)
                }
                AlgorithmChoice::CpuRayon => {
                    let solver = RandomizedSearchSolver::new(hash_variants);
                    solver.solve(&points, tolerance)
                }
            };
            let _ = tx.send(ComputationResult { squares, stats });
        });
    }

    /// Виконання натиснутої дії користувача
    pub fn execute_action(&mut self, action: ActionId) {
        match action {
            ActionId::SetAlgorithmGpu => {
                info!("UI action: Selected Vulkan dGPU compute algorithm (NVIDIA RTX 4050)");
                self.algorithm_choice = AlgorithmChoice::VulkanDgpu;
                self.status_message = "Обрано алгоритм: Vulkan dGPU (NVIDIA RTX 4050)".to_string();
            }
            ActionId::SetAlgorithmCpu => {
                info!("UI action: Selected CPU Rayon 12-thread algorithm");
                self.algorithm_choice = AlgorithmChoice::CpuRayon;
                self.status_message = "Обрано алгоритм: CPU Rayon (12 потоків)".to_string();
            }
            ActionId::RunAlg => {
                info!("UI action: Run algorithm triggered ({:?})", self.algorithm_choice);
                if !self.is_computing {
                    self.run_selected_algorithm();
                } else {
                    debug!("Algorithm is already running in background, ignoring duplicate trigger");
                }
            }
            ActionId::PickAndLoadFile => {
                info!("UI action: Opening file dialog to load points");
                let file_path = rfd::FileDialog::new()
                    .set_title("Оберіть файл точок (data.txt)")
                    .set_file_name("data.txt")
                    .add_filter("Текстові файли (*.txt)", &["txt"])
                    .pick_file();

                if let Some(path) = file_path {
                    if let Some(path_str) = path.to_str() {
                        info!("User selected file to load: '{}'", path_str);
                        self.load_file(path_str);
                    }
                } else {
                    debug!("File open dialog was cancelled by user");
                }
            }
            ActionId::PickAndSaveFile => {
                info!("UI action: Opening save file dialog for results");
                let file_path = rfd::FileDialog::new()
                    .set_title("Зберегти знайдені квадрати")
                    .set_file_name("Squares.txt")
                    .add_filter("Текстові файли (*.txt)", &["txt"])
                    .save_file();

                if let Some(path) = file_path {
                    if let Some(path_str) = path.to_str() {
                        info!("User selected destination file to save: '{}'", path_str);
                        self.save_file(path_str);
                    }
                } else {
                    debug!("File save dialog was cancelled by user");
                }
            }
            ActionId::ResetView => {
                info!("UI action: Reset view to fit all points");
                self.reset_view();
            }
            ActionId::ClearPoints => {
                if self.is_computing {
                    warn!("Cannot clear points while computation is active");
                    return;
                }
                info!("UI action: Clear all points and figures");
                self.clear();
            }
            ActionId::GenerateCustom => {
                if self.is_computing {
                    warn!("Cannot generate points while computation is active");
                    self.status_message = "Помилка: Зачекайте завершення поточних обчислень!".to_string();
                    return;
                }
                if let Ok(count) = self.point_input_str.trim().parse::<usize>() {
                    let safe_count = count.max(4); // Ліміт 50к повністю знято!
                    let squares_inject = (safe_count / 4).max(1);
                    info!(
                        "UI action: Generating custom dataset (requested: {}, count: {}, injected squares: {})",
                        count,
                        safe_count,
                        squares_inject
                    );
                    self.generate_points(safe_count, squares_inject);
                } else {
                    warn!(
                        "Invalid input string for point count: '{}'",
                        self.point_input_str
                    );
                }
            }
            ActionId::ApplyHashVariants => {
                if let Ok(val) = self.hash_input_str.trim().parse::<usize>() {
                    let target = val.max(1);
                    let prev = self.hash_variants;
                    self.hash_variants = target;
                    self.hash_input_str = target.to_string();
                    self.hash_cursor_pos = self.hash_input_str.len();
                    info!(
                        "Hash variants updated: {} -> {} (custom input)",
                        prev, self.hash_variants
                    );
                    self.status_message = format!("Встановлено {} хеш-варіантів", self.hash_variants);
                }
            }
            ActionId::DecHashVariants => {
                let prev = self.hash_variants;
                self.hash_variants = self.hash_variants.saturating_sub(10).max(1);
                self.hash_input_str = self.hash_variants.to_string();
                self.hash_cursor_pos = self.hash_input_str.len();
                info!("Hash variants decreased: {} -> {}", prev, self.hash_variants);
                self.status_message = format!("Кількість хеш-варіантів: {}", self.hash_variants);
            }
            ActionId::IncHashVariants => {
                let prev = self.hash_variants;
                self.hash_variants = self.hash_variants.saturating_add(10);
                self.hash_input_str = self.hash_variants.to_string();
                self.hash_cursor_pos = self.hash_input_str.len();
                info!("Hash variants increased: {} -> {}", prev, self.hash_variants);
                self.status_message = format!("Кількість хеш-варіантів: {}", self.hash_variants);
            }
            ActionId::SetHashVariants(val) => {
                let target = val.max(1);
                let prev = self.hash_variants;
                self.hash_variants = target;
                self.hash_input_str = target.to_string();
                self.hash_cursor_pos = self.hash_input_str.len();
                info!("Hash variants set: {} -> {}", prev, self.hash_variants);
                self.status_message = format!("Кількість хеш-варіантів: {}", self.hash_variants);
            }
        }
    }

    /// Обробка кліку миші
    pub fn handle_click(&mut self, x: i32, y: i32, _is_right: bool, window_width: i32) {
        let (ix, iy, iw, ih) = self.get_input_rect(window_width);
        if x >= ix && x < ix + iw && y >= iy && y < iy + ih {
            self.is_input_focused = true;
            self.is_hash_focused = false;
            let rel_x = x - (ix + 10);
            if rel_x <= 0 {
                self.input_cursor_pos = 0;
            } else {
                let char_idx = (rel_x + 4) / 8;
                self.input_cursor_pos = (char_idx as usize).min(self.point_input_str.len());
            }
            return;
        }

        let (hx, hy, hw, hh) = self.get_hash_input_rect(window_width);
        if x >= hx && x < hx + hw && y >= hy && y < hy + hh {
            self.is_hash_focused = true;
            self.is_input_focused = false;
            let rel_x = x - (hx + 10);
            if rel_x <= 0 {
                self.hash_cursor_pos = 0;
            } else {
                let char_idx = (rel_x + 4) / 8;
                self.hash_cursor_pos = (char_idx as usize).min(self.hash_input_str.len());
            }
            return;
        }

        self.is_input_focused = false;
        self.is_hash_focused = false;

        let buttons = self.get_buttons(window_width);
        for btn in buttons {
            if x >= btn.x && x < btn.x + btn.w && y >= btn.y && y < btn.y + btn.h {
                self.execute_action(btn.id);
                return;
            }
        }
    }

    /// Введення символу з клавіатури в текстове поле
    pub fn handle_char(&mut self, c: char) -> bool {
        if self.is_input_focused && c.is_ascii_digit() && self.point_input_str.len() < 10 {
            let pos = self.input_cursor_pos.min(self.point_input_str.len());
            self.point_input_str.insert(pos, c);
            self.input_cursor_pos += 1;
            return true;
        }
        if self.is_hash_focused && c.is_ascii_digit() && self.hash_input_str.len() < 8 {
            let pos = self.hash_cursor_pos.min(self.hash_input_str.len());
            self.hash_input_str.insert(pos, c);
            self.hash_cursor_pos += 1;
            if let Ok(v) = self.hash_input_str.parse::<usize>() {
                if v > 0 {
                    self.hash_variants = v;
                }
            }
            return true;
        }
        false
    }

    /// Видалення символу ліворуч (Backspace)
    pub fn handle_backspace(&mut self) -> bool {
        if self.is_input_focused && self.input_cursor_pos > 0 && !self.point_input_str.is_empty() {
            let pos = self.input_cursor_pos.min(self.point_input_str.len());
            self.point_input_str.remove(pos - 1);
            self.input_cursor_pos -= 1;
            return true;
        }
        if self.is_hash_focused && self.hash_cursor_pos > 0 && !self.hash_input_str.is_empty() {
            let pos = self.hash_cursor_pos.min(self.hash_input_str.len());
            self.hash_input_str.remove(pos - 1);
            self.hash_cursor_pos -= 1;
            if let Ok(v) = self.hash_input_str.parse::<usize>() {
                if v > 0 {
                    self.hash_variants = v;
                }
            }
            return true;
        }
        false
    }

    /// Видалення символу праворуч (Delete)
    pub fn handle_delete(&mut self) -> bool {
        if self.is_input_focused && self.input_cursor_pos < self.point_input_str.len() {
            self.point_input_str.remove(self.input_cursor_pos);
            return true;
        }
        if self.is_hash_focused && self.hash_cursor_pos < self.hash_input_str.len() {
            self.hash_input_str.remove(self.hash_cursor_pos);
            if let Ok(v) = self.hash_input_str.parse::<usize>() {
                if v > 0 {
                    self.hash_variants = v;
                }
            }
            return true;
        }
        false
    }

    /// Підтвердження введення за допомогою клавіші Enter
    pub fn handle_enter(&mut self) -> bool {
        if self.is_input_focused {
            self.is_input_focused = false;
            self.execute_action(ActionId::GenerateCustom);
            return true;
        }
        if self.is_hash_focused {
            self.is_hash_focused = false;
            self.execute_action(ActionId::ApplyHashVariants);
            return true;
        }
        false
    }

    /// Переміщення курсора вводу ліворуч
    pub fn move_cursor_left(&mut self) -> bool {
        if self.is_input_focused && self.input_cursor_pos > 0 {
            self.input_cursor_pos -= 1;
            return true;
        }
        if self.is_hash_focused && self.hash_cursor_pos > 0 {
            self.hash_cursor_pos -= 1;
            return true;
        }
        false
    }

    /// Переміщення курсора вводу праворуч
    pub fn move_cursor_right(&mut self) -> bool {
        if self.is_input_focused && self.input_cursor_pos < self.point_input_str.len() {
            self.input_cursor_pos += 1;
            return true;
        }
        if self.is_hash_focused && self.hash_cursor_pos < self.hash_input_str.len() {
            self.hash_cursor_pos += 1;
            return true;
        }
        false
    }

    /// Переміщення курсора вводу на початок рядка
    pub fn move_cursor_home(&mut self) -> bool {
        if self.is_input_focused && self.input_cursor_pos != 0 {
            self.input_cursor_pos = 0;
            return true;
        }
        if self.is_hash_focused && self.hash_cursor_pos != 0 {
            self.hash_cursor_pos = 0;
            return true;
        }
        false
    }

    /// Переміщення курсора вводу в кінець рядка
    pub fn move_cursor_end(&mut self) -> bool {
        if self.is_input_focused && self.input_cursor_pos != self.point_input_str.len() {
            self.input_cursor_pos = self.point_input_str.len();
            return true;
        }
        if self.is_hash_focused && self.hash_cursor_pos != self.hash_input_str.len() {
            self.hash_cursor_pos = self.hash_input_str.len();
            return true;
        }
        false
    }

    /// Завантаження точок з файлу
    pub fn load_file(&mut self, path: &str) {
        match FileManager::load_points(path) {
            Ok(pts) => {
                let count = pts.len();
                self.points = pts;
                self.reset_view();
                self.status_message = format!("Успішно завантажено {} точок з '{}'", count, path);
                self.run_selected_algorithm();
            }
            Err(e) => {
                error!("Failed to load points from '{}': {}", path, e);
                self.status_message = format!("Помилка читання файлу '{}': {}", path, e);
            }
        }
    }

    /// Збереження результатів у файл
    pub fn save_file(&mut self, path: &str) {
        match FileManager::save_results(path, &self.disjoint_squares, &self.stats) {
            Ok(()) => {
                self.status_message = format!(
                    "Результати ({} фігур) успішно збережено у '{}'",
                    self.disjoint_squares.len(),
                    path
                );
            }
            Err(e) => {
                error!("Failed to save results to '{}': {}", path, e);
                self.status_message = format!("Помилка запису файлу '{}': {}", path, e);
            }
        }
    }

    /// Перетворення екранних координат у світові
    #[inline]
    pub fn screen_to_world(&self, sx: i32, sy: i32) -> (f64, f64) {
        let wx = (sx as f64 - self.pan_x) / self.zoom;
        let wy = (sy as f64 - self.pan_y) / self.zoom;
        (wx, wy)
    }

    /// Перетворення світових координат у координати екрана
    #[inline]
    pub fn world_to_screen(&self, wx: f64, wy: f64) -> (i32, i32) {
        let sx = (wx * self.zoom + self.pan_x).round() as i32;
        let sy = (wy * self.zoom + self.pan_y).round() as i32;
        (sx, sy)
    }

    /// Генерація нового набору точок з гарантовано вбудованими повернутими квадратами
    pub fn generate_points(&mut self, count: usize, squares_to_inject: usize) {
        if self.is_computing {
            warn!("Cannot generate points while computation is active");
            self.status_message = "Помилка: Зачекайте завершення поточних обчислень!".to_string();
            return;
        }
        self.points = FileManager::generate_sample_dataset(count, 1000.0, 800.0, squares_to_inject);
        self.reset_view();
        self.status_message = format!(
            "Згенеровано {} точок на сітці 1000x800 з кроком 10",
            self.points.len()
        );

        // Зберігаємо згенеровані точки у вхідні файли
        let _ = FileManager::save_points("data.txt", &self.points);
        let _ = FileManager::save_points("input.txt", &self.points);

        self.run_selected_algorithm();
    }

    /// Скидання положення та масштабу полотна до початкового стану
    pub fn reset_view(&mut self) {
        self.auto_fit_view();
    }

    /// Автоматичне масштабування та центрування точок на екрані з кратним дискретним зумом
    pub fn auto_fit_view(&mut self) {
        if self.points.is_empty() {
            self.pan_x = 40.0;
            self.pan_y = 40.0;
            self.zoom = 1.0;
            return;
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for p in &self.points {
            if p.x < min_x { min_x = p.x; }
            if p.x > max_x { max_x = p.x; }
            if p.y < min_y { min_y = p.y; }
            if p.y > max_y { max_y = p.y; }
        }

        let span_x = (max_x - min_x).max(50.0);
        let span_y = (max_y - min_y).max(50.0);

        let canvas_w = 750.0;
        let canvas_h = 650.0;

        let zoom_x = (canvas_w - 80.0) / span_x;
        let zoom_y = (canvas_h - 80.0) / span_y;
        let raw_zoom = zoom_x.min(zoom_y);

        // Знаходимо найбільший фіксований дискретний рівень зуму, який повністю поміщає вміст
        let mut fitted = ZOOM_LEVELS[0];
        for &lvl in ZOOM_LEVELS {
            if lvl <= raw_zoom + 0.001 {
                fitted = lvl;
            } else {
                break;
            }
        }

        self.zoom = fitted.clamp(0.25, 5.0);
        self.pan_x = 40.0 - min_x * self.zoom;
        self.pan_y = 40.0 - min_y * self.zoom;
        debug!(
            "Auto-fitted canvas view: zoom={:.2}x, pan=[{:.1}, {:.1}] for {} points",
            self.zoom,
            self.pan_x,
            self.pan_y,
            self.points.len()
        );
    }

    /// Очищення полотна від усіх точок та результатів
    pub fn clear(&mut self) {
        info!("Clearing canvas: removing all points, candidate squares, and disjoint solutions");
        self.points.clear();
        self.candidate_squares.clear();
        self.disjoint_squares.clear();
        self.square_points_mask.clear();
        self.stats = AlgorithmStats::default();
        self.status_message = "Полотно очищено. Додайте нові точки.".to_string();
    }

    /// Головний метод візуалізації вмісту полотна
    pub fn render(&mut self, canvas: &mut Canvas) {
        self.anim_tick = self.anim_tick.wrapping_add(1);

        let width = canvas.width as i32;
        let height = canvas.height as i32;

        let sidebar_x = self.sidebar_x(width);
        let status_bar_h = 24;
        let canvas_max_y = height - status_bar_h;

        // 1. Очищення полотна темним фоном
        canvas.clear(Color::rgb(13, 17, 23));

        // 2. Відмальовування координатної сітки та осей у світових координатах
        self.render_grid(canvas, sidebar_x, canvas_max_y);
        self.render_axes(canvas, sidebar_x, canvas_max_y);

        // 3. Відмальовування побудованих незалежних квадратів
        self.render_squares(canvas, sidebar_x, canvas_max_y);

        // 4. Швидке відмальовування точок
        self.render_points(canvas, sidebar_x, canvas_max_y);

        // 5. Відмальовування правої панелі керування та статистики
        self.render_sidebar(canvas, sidebar_x, width, canvas_max_y);

        // 6. Відмальовування нижнього статус-бару
        self.render_status_bar(canvas, width, height);

        // 7. Інформаційний банер під час активного фонового обчислення
        if self.is_computing {
            let banner_w = 340;
            let banner_h = 44;
            let bx = (sidebar_x - banner_w) / 2;
            let by = 20;
            let pulse = ((self.anim_tick as f64 * 0.15).sin() * 25.0).round() as i32;
            let bg_r = (25 + pulse).clamp(0, 255) as u8;
            let bg_g = (60 + pulse).clamp(0, 255) as u8;
            let bg_b = (110 + pulse).clamp(0, 255) as u8;

            canvas.draw_rect(bx, by, banner_w, banner_h, Color::rgb(bg_r, bg_g, bg_b), true);
            canvas.draw_rect(bx, by, banner_w, banner_h, Color::rgb(88, 166, 255), false);
            canvas.draw_text(
                bx + 14,
                by + 14,
                "Обчислення незалежних квадратiв...",
                Color::rgb(255, 255, 255),
                1,
            );
        }
    }

    /// Відмальовування координатної сітки у світовому просторі
    fn render_grid(&self, canvas: &mut Canvas, max_x: i32, max_y: i32) {
        let zoom = if self.zoom.is_finite() && self.zoom > 0.001 { self.zoom } else { 1.0 };
        let pan_x = self.pan_x;
        let pan_y = self.pan_y;

        let left_w = (0.0 - pan_x) / zoom;
        let right_w = (max_x as f64 - pan_x) / zoom;
        let top_w = (0.0 - pan_y) / zoom;
        let bottom_w = (max_y as f64 - pan_y) / zoom;

        let min_w_x = left_w.min(right_w);
        let max_w_x = left_w.max(right_w);
        let min_w_y = top_w.min(bottom_w);
        let max_w_y = top_w.max(bottom_w);

        let target_screen_px = 60.0;
        let raw_step = (target_screen_px / zoom).max(0.01);

        let exponent = raw_step.log10().floor();
        let fraction = raw_step / 10.0_f64.powf(exponent);

        let nice_factor = if fraction < 1.5 {
            1.0
        } else if fraction < 3.5 {
            2.0
        } else if fraction < 7.5 {
            5.0
        } else {
            10.0
        };

        let world_step = nice_factor * 10.0_f64.powf(exponent);
        if !world_step.is_finite() || world_step <= 0.0 {
            return;
        }

        let first_x = (min_w_x / world_step).floor() * world_step;
        let first_y = (min_w_y / world_step).floor() * world_step;

        let grid_color = Color::rgb(25, 30, 40);
        let c_u32 = grid_color.to_bgra_u32();
        let width = canvas.width;

        // Вертикальні лінії сітки (прямий запис у пам'ять)
        let mut wx = first_x;
        while wx <= max_w_x + world_step {
            let sx = (wx * zoom + pan_x).round() as i32;
            if sx >= 0 && sx < max_x {
                for y in 0..max_y {
                    canvas.pixels[(y as usize) * width + sx as usize] = c_u32;
                }
            }
            wx += world_step;
        }

        // Горизонтальні лінії сітки (швидкий slice fill)
        let mut wy = first_y;
        while wy <= max_w_y + world_step {
            let sy = (wy * zoom + pan_y).round() as i32;
            if sy >= 0 && sy < max_y {
                let row_start = (sy as usize) * width;
                canvas.pixels[row_start..row_start + max_x as usize].fill(c_u32);
            }
            wy += world_step;
        }
    }

    /// Відмальовування осей X та Y
    fn render_axes(&self, canvas: &mut Canvas, max_x: i32, max_y: i32) {
        let (zero_x, zero_y) = self.world_to_screen(0.0, 0.0);
        let axis_color = Color::rgb(70, 80, 95);
        let c_u32 = axis_color.to_bgra_u32();
        let width = canvas.width;

        // Вісь Y (вертикальна, товщина 2)
        if zero_x >= 0 && zero_x < max_x {
            for y in 0..max_y {
                let offset = (y as usize) * width + zero_x as usize;
                canvas.pixels[offset] = c_u32;
                if zero_x + 1 < max_x {
                    canvas.pixels[offset + 1] = c_u32;
                }
            }
        }

        // Вісь X (горизонтальна, товщина 2)
        if zero_y >= 0 && zero_y < max_y {
            let row_start = (zero_y as usize) * width;
            canvas.pixels[row_start..row_start + max_x as usize].fill(c_u32);
            if zero_y + 1 < max_y {
                let next_row = row_start + width;
                canvas.pixels[next_row..next_row + max_x as usize].fill(c_u32);
            }
        }
    }

    /// Відмальовування знайдених квадратів із AABB відсіканням за межами екрана
    fn render_squares(&self, canvas: &mut Canvas, max_x: i32, max_y: i32) {
        let count = self.disjoint_squares.len();
        if count == 0 {
            return;
        }

        let border_colors = [
            Color::rgb(88, 166, 255),  // Блакитний
            Color::rgb(63, 185, 80),   // Зелений
            Color::rgb(240, 136, 62),  // Помаранчевий
            Color::rgb(210, 153, 34),  // Золотистий
            Color::rgb(219, 97, 162),  // Рожевий
            Color::rgb(187, 128, 255), // Фіолетовий
        ];

        let show_labels = count <= 80;

        let zoom = self.zoom;
        let pan_x = self.pan_x;
        let pan_y = self.pan_y;

        // Межі видимого світового простору (Viewport Frustum)
        let view_min_wx = (0.0 - pan_x) / zoom;
        let view_max_wx = (max_x as f64 - pan_x) / zoom;
        let view_min_wy = (0.0 - pan_y) / zoom;
        let view_max_wy = (max_y as f64 - pan_y) / zoom;

        for (idx, sq) in self.disjoint_squares.iter().enumerate() {
            // AABB відсікання квадратів за межами екрана (миттєве відкидання без викликів draw_line)
            let sq_min_x = sq.points[0].x.min(sq.points[1].x).min(sq.points[2].x).min(sq.points[3].x);
            let sq_max_x = sq.points[0].x.max(sq.points[1].x).max(sq.points[2].x).max(sq.points[3].x);
            let sq_min_y = sq.points[0].y.min(sq.points[1].y).min(sq.points[2].y).min(sq.points[3].y);
            let sq_max_y = sq.points[0].y.max(sq.points[1].y).max(sq.points[2].y).max(sq.points[3].y);

            if sq_max_x < view_min_wx || sq_min_x > view_max_wx || sq_max_y < view_min_wy || sq_min_y > view_max_wy {
                continue;
            }

            let color = border_colors[idx % border_colors.len()];

            let p0 = self.world_to_screen(sq.points[0].x, sq.points[0].y);
            let p1 = self.world_to_screen(sq.points[1].x, sq.points[1].y);
            let p2 = self.world_to_screen(sq.points[2].x, sq.points[2].y);
            let p3 = self.world_to_screen(sq.points[3].x, sq.points[3].y);

            // Малюємо 4 зовнішні сторони контуру квадрата
            canvas.draw_line(p0.0, p0.1, p1.0, p1.1, color, 2);
            canvas.draw_line(p1.0, p1.1, p2.0, p2.1, color, 2);
            canvas.draw_line(p2.0, p2.1, p3.0, p3.1, color, 2);
            canvas.draw_line(p3.0, p3.1, p0.0, p0.1, color, 2);

            if show_labels {
                let (cx, cy) = sq.center();
                let (scx, scy) = self.world_to_screen(cx, cy);

                if scx >= 20 && scx < max_x - 30 && scy >= 10 && scy < max_y - 20 {
                    let label = format!("#{}", idx + 1);
                    let lx = scx - (label.len() as i32 * 4);
                    let ly = scy - 4;
                    canvas.draw_rect(lx - 2, ly - 2, (label.len() as i32 * 8) + 4, 12, Color::rgb(13, 17, 23), true);
                    canvas.draw_text(lx, ly, &label, Color::rgb(255, 255, 255), 1);
                }
            }
        }
    }

    /// Відмальовування точок із відсіканням невидимих точок за межами екрана
    fn render_points(&self, canvas: &mut Canvas, max_x: i32, max_y: i32) {
        let default_color = Color::rgb(139, 148, 158);
        let square_color = Color::rgb(255, 215, 0);

        let zoom = self.zoom;
        let pan_x = self.pan_x;
        let pan_y = self.pan_y;
        let total = self.points.len();
        let show_text = total <= 200;

        let mask_len = self.square_points_mask.len();

        let view_min_wx = (-10.0 - pan_x) / zoom;
        let view_max_wx = (max_x as f64 + 10.0 - pan_x) / zoom;
        let view_min_wy = (-10.0 - pan_y) / zoom;
        let view_max_wy = (max_y as f64 + 10.0 - pan_y) / zoom;

        for (idx, pt) in self.points.iter().enumerate() {
            // Швидке відсікання точок, що не потрапляють у видиму область
            if pt.x < view_min_wx || pt.x > view_max_wx || pt.y < view_min_wy || pt.y > view_max_wy {
                continue;
            }

            let sx = (pt.x * zoom + pan_x).round() as i32;
            let sy = (pt.y * zoom + pan_y).round() as i32;

            if sx >= 0 && sx < max_x && sy >= 0 && sy < max_y {
                let in_square = idx < mask_len && self.square_points_mask[idx];
                let color = if in_square { square_color } else { default_color };

                canvas.draw_point_marker(sx, sy, color, if in_square { 3 } else { 2 });

                if show_text && sx >= 0 && sx < max_x - 40 && sy >= 0 && sy < max_y - 12 {
                    canvas.draw_text(sx + 5, sy - 4, &format!("P{}", pt.id), Color::rgb(180, 190, 205), 1);
                }
            }
        }
    }

    /// Відмальовування правої панелі
    fn render_sidebar(&self, canvas: &mut Canvas, sx: i32, max_x: i32, max_y: i32) {
        let w = max_x - sx;
        canvas.draw_rect(sx, 0, w, max_y, Color::rgb(13, 17, 23), true);
        canvas.draw_line(sx, 0, sx, max_y, Color::rgb(48, 54, 61), 2);

        let pad_x = sx + 15;
        let card_w = w - 30;

        // Заголовок курсової роботи
        canvas.draw_text(pad_x, 15, "КУРСОВА РОБОТА: АСД", Color::rgb(88, 166, 255), 2);
        let gpu_label = format!("Vulkan: {} ({})", self.gpu_name, if self.is_igpu { "iGPU" } else { "dGPU" });
        canvas.draw_text_clipped(pad_x, 52, &gpu_label, Color::rgb(63, 185, 80), 1, max_x - 10);

        canvas.draw_line(pad_x, 70, max_x - 15, 70, Color::rgb(48, 54, 61), 1);

        // 1. Інформаційна картка обраного алгоритму розв'язання
        canvas.draw_rect(pad_x, 80, card_w, 70, Color::rgb(22, 27, 34), true);
        canvas.draw_rect(pad_x, 80, card_w, 70, Color::rgb(48, 54, 61), false);
        canvas.draw_text_clipped(pad_x + 8, 86, "ОБРАНИЙ АЛГОРИТМ:", Color::rgb(88, 166, 255), 1, max_x - 10);
        let alg_title = match self.algorithm_choice {
            AlgorithmChoice::VulkanDgpu => "dGPU (NVIDIA RTX 4050 - Warps)",
            AlgorithmChoice::CpuRayon => "CPU Rayon (12 апаратних потоків)",
        };
        let alg_color = match self.algorithm_choice {
            AlgorithmChoice::VulkanDgpu => Color::rgb(63, 185, 80),
            AlgorithmChoice::CpuRayon => Color::rgb(88, 166, 255),
        };
        canvas.draw_text_clipped(pad_x + 8, 102, alg_title, alg_color, 1, max_x - 10);

        // 2. Секція: ХЕШ ВАРІАНТИ (МУЛЬТИСТАРТ)
        canvas.draw_line(pad_x, 156, max_x - 15, 156, Color::rgb(48, 54, 61), 1);
        canvas.draw_text_clipped(pad_x, 162, "ХЕШ ВАРIАНТИ (МУЛЬТИСТАРТ):", Color::rgb(240, 246, 252), 1, max_x - 10);

        // Поле вводу кількості хеш варіантів
        let (hx, hy, hw, hh) = self.get_hash_input_rect(max_x);
        let hash_border = if self.is_hash_focused {
            Color::rgb(88, 166, 255)
        } else {
            Color::rgb(60, 70, 85)
        };
        canvas.draw_rect(hx, hy, hw, hh, Color::rgb(22, 27, 34), true);
        canvas.draw_rect(hx, hy, hw, hh, hash_border, false);

        if self.hash_input_str.is_empty() {
            canvas.draw_text(hx + 10, hy + 9, "Вар...", Color::rgb(110, 120, 135), 1);
        } else {
            canvas.draw_text(hx + 10, hy + 9, &self.hash_input_str, Color::rgb(255, 255, 255), 1);
        }

        // Миготливий текстовий курсор для поля хеш варіантів
        if self.is_hash_focused && (self.anim_tick / 20) % 2 == 0 {
            let cur_pos = self.hash_cursor_pos.min(self.hash_input_str.len()) as i32;
            let cursor_x = hx + 10 + cur_pos * 8;
            canvas.draw_line(cursor_x, hy + 5, cursor_x, hy + hh - 5, Color::rgb(88, 166, 255), 2);
        }

        let active_hash_lbl = format!("Активно: {} вар. (дефолт 40)", self.hash_variants);
        canvas.draw_text_clipped(pad_x, 214, &active_hash_lbl, Color::rgb(139, 148, 158), 1, max_x - 10);

        // 3. Секція: ОСНОВНІ ДІЇ
        canvas.draw_line(pad_x, 234, max_x - 15, 234, Color::rgb(48, 54, 61), 1);
        canvas.draw_text(pad_x, 238, "ОСНОВНI ДIЇ:", Color::rgb(240, 246, 252), 1);

        // 4. Секція: ГЕНЕРАЦІЯ ТОЧОК
        canvas.draw_line(pad_x, 356, max_x - 15, 356, Color::rgb(48, 54, 61), 1);
        canvas.draw_text(pad_x, 360, "ГЕНЕРАЦIЯ ТОЧОК (введiть N):", Color::rgb(240, 246, 252), 1);

        // Поле вводу кількості точок
        let (ix, iy, iw, ih) = self.get_input_rect(max_x);
        let input_bg = Color::rgb(22, 27, 34);
        let input_border = if self.is_input_focused {
            Color::rgb(88, 166, 255)
        } else {
            Color::rgb(60, 70, 85)
        };

        canvas.draw_rect(ix, iy, iw, ih, input_bg, true);
        canvas.draw_rect(ix, iy, iw, ih, input_border, false);

        if self.point_input_str.is_empty() {
            canvas.draw_text(ix + 10, iy + 9, "N...", Color::rgb(110, 120, 135), 1);
        } else {
            canvas.draw_text(ix + 10, iy + 9, &self.point_input_str, Color::rgb(255, 255, 255), 1);
        }

        // Миготливий текстовий курсор
        if self.is_input_focused && (self.anim_tick / 20) % 2 == 0 {
            let cur_pos = self.input_cursor_pos.min(self.point_input_str.len()) as i32;
            let cursor_x = ix + 10 + cur_pos * 8;
            canvas.draw_line(cursor_x, iy + 5, cursor_x, iy + ih - 5, Color::rgb(88, 166, 255), 2);
        }

        canvas.draw_text(pad_x, 410, "Введiть число та натиснiть Enter", Color::rgb(110, 120, 135), 1);

        // Малювання кнопок
        let buttons = self.get_buttons(max_x);
        for btn in buttons {
            let is_hovered = self.mouse_screen_x >= btn.x
                && self.mouse_screen_x < btn.x + btn.w
                && self.mouse_screen_y >= btn.y
                && self.mouse_screen_y < btn.y + btn.h;

            let mut bg = btn.bg_color;
            if is_hovered {
                bg = Color::rgba(
                    bg.r.saturating_add(30),
                    bg.g.saturating_add(30),
                    bg.b.saturating_add(30),
                    255,
                );
            }

            canvas.draw_rect(btn.x, btn.y, btn.w, btn.h, bg, true);
            canvas.draw_rect(
                btn.x,
                btn.y,
                btn.w,
                btn.h,
                if is_hovered { Color::rgb(255, 255, 255) } else { Color::rgb(70, 80, 95) },
                false,
            );
            let text_w = (btn.label.len() as i32) * 8;
            let text_x = btn.x + ((btn.w - text_w) / 2).max(4);
            canvas.draw_text(text_x, btn.y + (btn.h - 8) / 2, btn.label, Color::rgb(255, 255, 255), 1);
        }

        // 5. Секція: СТАТИСТИКА ТА ЧАС
        canvas.draw_line(pad_x, 430, max_x - 15, 430, Color::rgb(48, 54, 61), 1);
        canvas.draw_text_clipped(pad_x, 438, "СТАТИСТИКА ТА ЧАС РОБОТИ:", Color::rgb(240, 246, 252), 1, max_x - 10);

        let stat_y = 458;
        let line_h = 16;
        canvas.draw_text_clipped(pad_x, stat_y, &format!("Вхiдних точок: {}", self.points.len()), Color::rgb(139, 148, 158), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, stat_y + line_h, &format!("Кандидатiв у квадрати: {}", self.stats.candidates_found), Color::rgb(139, 148, 158), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, stat_y + line_h * 2, &format!("Хеш варiантiв: {} (дефолт 40)", self.hash_variants), Color::rgb(88, 166, 255), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, stat_y + line_h * 3, &format!("ПОБУДОВАНО ФIГУР: {}", self.disjoint_squares.len()), Color::rgb(46, 160, 67), 1, max_x - 10);

        canvas.draw_text_clipped(pad_x, stat_y + line_h * 4, &format!("Час детекцiї: {:.3} мс", self.stats.detection_time_ms), Color::rgb(88, 166, 255), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, stat_y + line_h * 5, &format!("Час вибору: {:.3} мс", self.stats.selection_time_ms), Color::rgb(88, 166, 255), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, stat_y + line_h * 6, &format!("ЗАГАЛЬНИЙ ЧАС: {:.3} мс", self.stats.total_time_ms), Color::rgb(255, 180, 0), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, stat_y + line_h * 7, &format!("Оцiнка пам'ятi: {:.1} КБ", self.stats.memory_kb), Color::rgb(139, 148, 158), 1, max_x - 10);

        // 6. Інструкція користувача
        let ctrl_y = stat_y + line_h * 8 + 6;
        canvas.draw_line(pad_x, ctrl_y - 6, max_x - 15, ctrl_y - 6, Color::rgb(48, 54, 61), 1);
        canvas.draw_text_clipped(pad_x, ctrl_y, "КЕРУВАННЯ:", Color::rgb(240, 246, 252), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, ctrl_y + 16, "* Перетягування мишею: Pan", Color::rgb(139, 148, 158), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, ctrl_y + 32, "* Колiщатко: Zoom (масштаб)", Color::rgb(139, 148, 158), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, ctrl_y + 48, "* Вiдцентрувати: скинути вид", Color::rgb(139, 148, 158), 1, max_x - 10);
        canvas.draw_text_clipped(pad_x, ctrl_y + 64, "* Генерацiя: поле N та Enter", Color::rgb(139, 148, 158), 1, max_x - 10);
    }

    /// Відмальовування нижнього рядка стану (Status Bar)
    fn render_status_bar(&self, canvas: &mut Canvas, max_x: i32, height: i32) {
        let bar_h = 24;
        let bar_y = height - bar_h;
        let bar_bg = Color::rgb(13, 17, 23);
        let border_color = Color::rgb(48, 54, 61);

        canvas.draw_rect(0, bar_y, max_x, bar_h, bar_bg, true);
        canvas.draw_line(0, bar_y, max_x, bar_y, border_color, 1);

        let (wx, wy) = self.screen_to_world(self.mouse_screen_x, self.mouse_screen_y);
        let coord_text = format!("X: {:.1}  Y: {:.1}  Zoom: {}", wx, wy, format_zoom(self.zoom));
        let coord_w = (coord_text.len() as i32) * 8;
        let coord_x = max_x - coord_w - 12;
        let text_y = bar_y + (bar_h - 8) / 2;

        if coord_x > 200 {
            let sep_x = coord_x - 12;
            canvas.draw_line(sep_x, bar_y + 4, sep_x, bar_y + bar_h - 4, border_color, 1);

            canvas.draw_text_clipped(coord_x, text_y, &coord_text, Color::rgb(139, 148, 158), 1, max_x - 4);

            let avail_w = sep_x - 20;
            let max_chars = (avail_w / 8).max(0) as usize;
            let char_count = self.status_message.chars().count();

            if char_count > max_chars && max_chars > 3 {
                let truncated: String = self.status_message.chars().take(max_chars - 3).collect();
                let display_msg = format!("{}...", truncated);
                canvas.draw_text_clipped(10, text_y, &display_msg, Color::rgb(56, 139, 253), 1, sep_x - 6);
            } else {
                canvas.draw_text_clipped(10, text_y, &self.status_message, Color::rgb(56, 139, 253), 1, sep_x - 6);
            }
        } else {
            canvas.draw_text_clipped(10, text_y, &self.status_message, Color::rgb(56, 139, 253), 1, max_x - 10);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_zoom_stepping_and_formatting() {
        assert_eq!(format_zoom(1.0), "1x");
        assert_eq!(format_zoom(2.0), "2x");
        assert_eq!(format_zoom(0.5), "0.5x");
        assert_eq!(format_zoom(0.75), "0.75x");
        assert_eq!(format_zoom(1.25), "1.25x");
        assert_eq!(format_zoom(2.5), "2.5x");

        let z1 = 1.0;
        let z_up = next_zoom_in(z1);
        assert_eq!(z_up, 1.25);
        let z_up2 = next_zoom_in(z_up);
        assert_eq!(z_up2, 1.5);

        let z_down = next_zoom_out(z1);
        assert_eq!(z_down, 0.75);
        let z_down2 = next_zoom_out(z_down);
        assert_eq!(z_down2, 0.5);
    }

    #[test]
    fn test_cursor_navigation_and_editing() {
        let mut state = UiState::new();
        state.is_input_focused = true;
        state.point_input_str = "123".to_string();
        state.input_cursor_pos = 3;

        state.move_cursor_left();
        assert_eq!(state.input_cursor_pos, 2);

        state.handle_char('9');
        assert_eq!(state.point_input_str, "1293");
        assert_eq!(state.input_cursor_pos, 3);

        state.handle_backspace();
        assert_eq!(state.point_input_str, "123");
        assert_eq!(state.input_cursor_pos, 2);

        state.move_cursor_home();
        assert_eq!(state.input_cursor_pos, 0);

        state.move_cursor_end();
        assert_eq!(state.input_cursor_pos, 3);
    }

    #[test]
    fn test_canvas_text_clipping() {
        let mut canvas = Canvas::new(100, 50);
        canvas.clear(Color::rgb(0, 0, 0));

        canvas.draw_text_clipped(0, 0, "ДОВГИЙ_ТЕКСТ_ДЛЯ_ОБРІЗКИ_ПО_ШИРИНІ", Color::rgb(255, 255, 255), 1, 24);

        for y in 0..50 {
            for x in 24..100 {
                let pixel = canvas.pixels[y * canvas.width + x];
                assert_eq!(pixel, Color::rgb(0, 0, 0).to_bgra_u32());
            }
        }
    }

    #[test]
    fn test_grid_and_axes_locking_under_zoom_and_pan() {
        let mut state = UiState::new();
        let mut canvas = Canvas::new(800, 600);

        for zoom in [0.25, 0.5, 1.0, 2.5, 5.0] {
            for pan in [-200.0, -50.0, 0.0, 150.0, 300.0] {
                state.zoom = zoom;
                state.pan_x = pan;
                state.pan_y = pan;

                canvas.clear(Color::rgb(0, 0, 0));
                state.render_grid(&mut canvas, 600, 600);

                let (zero_x, zero_y) = state.world_to_screen(0.0, 0.0);
                if zero_x >= 0 && zero_x < 600 && zero_y >= 0 && zero_y < 600 {
                    let pixel = canvas.pixels[zero_y as usize * canvas.width + zero_x as usize];
                    assert_ne!(
                        pixel,
                        Color::rgb(0, 0, 0).to_bgra_u32(),
                        "Перетин осей при zero_x={}, zero_y={} повинен відмальовуватися для zoom={}, pan={}",
                        zero_x, zero_y, zoom, pan
                    );
                }
            }
        }
    }

    #[test]
    fn test_hash_variants_selection_and_stepping() {
        let mut state = UiState::new();
        // Перевірка значення за замовчуванням: 40
        assert_eq!(state.hash_variants, 40);

        // Довільне введення хеш варіантів
        state.hash_input_str = "125".to_string();
        state.execute_action(ActionId::ApplyHashVariants);
        assert_eq!(state.hash_variants, 125);

        // Встановлення через SetHashVariants
        state.execute_action(ActionId::SetHashVariants(500));
        assert_eq!(state.hash_variants, 500);

        // Крок +10
        state.execute_action(ActionId::IncHashVariants);
        assert_eq!(state.hash_variants, 510);

        // Крок -10
        state.execute_action(ActionId::DecHashVariants);
        assert_eq!(state.hash_variants, 500);
    }

    #[test]
    fn test_algorithm_selection_switching() {
        let mut state = UiState::new();
        assert_eq!(state.algorithm_choice, AlgorithmChoice::VulkanDgpu);

        state.execute_action(ActionId::SetAlgorithmCpu);
        assert_eq!(state.algorithm_choice, AlgorithmChoice::CpuRayon);

        state.execute_action(ActionId::SetAlgorithmGpu);
        assert_eq!(state.algorithm_choice, AlgorithmChoice::VulkanDgpu);
    }
}
