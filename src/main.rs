#![allow(non_snake_case)] // Allows CamelCase crate name ASDKyrsova

pub mod algorithms;
pub mod geometry;
pub mod gui;
pub mod head;
pub mod io;
pub mod spatial;

fn main() {
    // Ініціалізація системи логування з часовими мітками у мілісекундах
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    // Налаштовуємо глобальний пул Rayon на максимальну кількість доступних апаратних ядер/потоків
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();

    log::info!(
        "Starting ASDKyrsova on {} Rayon hardware worker threads...",
        num_threads
    );
    head::run();
}
