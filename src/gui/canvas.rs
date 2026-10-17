use super::font::{get_glyph, CHAR_HEIGHT, CHAR_WIDTH};
use rayon::prelude::*;

/// 32-бітний колір у форматі RGBA
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    pub const fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as u8,
            g: ((hex >> 8) & 0xFF) as u8,
            b: (hex & 0xFF) as u8,
            a: 255,
        }
    }

    /// Конвертація у 32-бітний піксель B8G8R8A8 (стандарт для Windows Vulkan Swapchain)
    #[inline(always)]
    pub fn to_bgra_u32(self) -> u32 {
        ((self.a as u32) << 24)
            | ((self.r as u32) << 16)
            | ((self.g as u32) << 8)
            | (self.b as u32)
    }
}

/// 2D растрове полотно для малювання геометричних примітивів та елементів інтерфейсу
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
}

#[inline(always)]
fn compute_outcode(x: i32, y: i32, w: i32, h: i32) -> u8 {
    let mut code = 0;
    if x < 0 {
        code |= 1; // Зліва
    } else if x >= w {
        code |= 2; // Справа
    }
    if y < 0 {
        code |= 4; // Знизу
    } else if y >= h {
        code |= 8; // Зверху
    }
    code
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            pixels: vec![0xFF000000; size],
        }
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width != width || self.height != height {
            let size = width * height;
            log::debug!("Canvas frame buffer resized: {}x{} ({} pixels)", width, height, size);
            self.width = width;
            self.height = height;
            self.pixels.resize(size, 0xFF000000);
        }
    }

    /// Багатопотокове швидке очищення буфера на CPU через Rayon
    #[inline]
    pub fn clear(&mut self, color: Color) {
        let c_u32 = color.to_bgra_u32();
        self.pixels.par_chunks_mut(16384).for_each(|chunk| {
            chunk.fill(c_u32);
        });
    }

    /// Встановлення окремого пікселя з альфа-змішуванням
    #[inline]
    pub fn set_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
            let idx = (y as usize) * self.width + (x as usize);
            if color.a == 255 {
                self.pixels[idx] = color.to_bgra_u32();
            } else if color.a > 0 {
                let bg = self.pixels[idx];
                let bg_b = bg & 0xFF;
                let bg_g = (bg >> 8) & 0xFF;
                let bg_r = (bg >> 16) & 0xFF;

                let alpha = color.a as u32;
                let inv_alpha = 255 - alpha;

                let out_b = (color.b as u32 * alpha + bg_b * inv_alpha) / 255;
                let out_g = (color.g as u32 * alpha + bg_g * inv_alpha) / 255;
                let out_r = (color.r as u32 * alpha + bg_r * inv_alpha) / 255;

                self.pixels[idx] = 0xFF000000 | (out_r << 16) | (out_g << 8) | out_b;
            }
        }
    }

    /// Швидке малювання маркера точки безпосереднім записом у пам'ять
    #[inline]
    pub fn draw_point_marker(&mut self, x: i32, y: i32, color: Color, size: i32) {
        let c_u32 = color.to_bgra_u32();
        let half = size / 2;
        let x0 = (x - half).max(0);
        let x1 = (x + half).min(self.width as i32 - 1);
        let y0 = (y - half).max(0);
        let y1 = (y + half).min(self.height as i32 - 1);

        for py in y0..=y1 {
            let row_offset = (py as usize) * self.width;
            for px in x0..=x1 {
                self.pixels[row_offset + px as usize] = c_u32;
            }
        }
    }

    /// Малювання прямокутника (заповненого або контурного)
    pub fn draw_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color, filled: bool) {
        if w <= 0 || h <= 0 {
            return;
        }

        if filled {
            let x_start = x.max(0);
            let y_start = y.max(0);
            let x_end = (x + w).min(self.width as i32);
            let y_end = (y + h).min(self.height as i32);

            let c_u32 = color.to_bgra_u32();
            if color.a == 255 {
                for cy in y_start..y_end {
                    let row_start = (cy as usize) * self.width;
                    let slice = &mut self.pixels[row_start + x_start as usize..row_start + x_end as usize];
                    slice.fill(c_u32);
                }
            } else {
                for cy in y_start..y_end {
                    for cx in x_start..x_end {
                        self.set_pixel(cx, cy, color);
                    }
                }
            }
        } else {
            self.draw_line(x, y, x + w - 1, y, color, 1);
            self.draw_line(x + w - 1, y, x + w - 1, y + h - 1, color, 1);
            self.draw_line(x + w - 1, y + h - 1, x, y + h - 1, color, 1);
            self.draw_line(x, y + h - 1, x, y, color, 1);
        }
    }

    /// Високопродуктивне малювання лінії за алгоритмом Коена-Сазерленда (відсікання) та Брезенхема.
    /// Відсікає невидимі ділянки за O(1), усуваючи просідання FPS при масштабуванні полотна.
    pub fn draw_line(
        &mut self,
        mut x0: i32,
        mut y0: i32,
        mut x1: i32,
        mut y1: i32,
        color: Color,
        thickness: i32,
    ) {
        let w = self.width as i32;
        let h = self.height as i32;

        let mut code0 = compute_outcode(x0, y0, w, h);
        let mut code1 = compute_outcode(x1, y1, w, h);

        // Алгоритм відсікання відрізків Коена-Сазерленда
        loop {
            if (code0 | code1) == 0 {
                // Обидва кінці всередині екрана
                break;
            } else if (code0 & code1) != 0 {
                // Лінія повністю за межами екрана — миттєве відкидання без циклів
                return;
            } else {
                let outcode = if code0 != 0 { code0 } else { code1 };
                let mut x = 0;
                let mut y = 0;

                if (outcode & 8) != 0 {
                    // Перетин з верхньою межею (y >= h)
                    x = x0 + ((x1 - x0) as i64 * (h - 1 - y0) as i64 / (y1 - y0) as i64) as i32;
                    y = h - 1;
                } else if (outcode & 4) != 0 {
                    // Перетин з нижньою межею (y < 0)
                    x = x0 + ((x1 - x0) as i64 * (0 - y0) as i64 / (y1 - y0) as i64) as i32;
                    y = 0;
                } else if (outcode & 2) != 0 {
                    // Перетин з правою межею (x >= w)
                    y = y0 + ((y1 - y0) as i64 * (w - 1 - x0) as i64 / (x1 - x0) as i64) as i32;
                    x = w - 1;
                } else if (outcode & 1) != 0 {
                    // Перетин з лівою межею (x < 0)
                    y = y0 + ((y1 - y0) as i64 * (0 - x0) as i64 / (x1 - x0) as i64) as i32;
                    x = 0;
                }

                if outcode == code0 {
                    x0 = x;
                    y0 = y;
                    code0 = compute_outcode(x0, y0, w, h);
                } else {
                    x1 = x;
                    y1 = y;
                    code1 = compute_outcode(x1, y1, w, h);
                }
            }
        }

        // Тепер кінці гарантовано знаходяться в межах [0, w-1] x [0, h-1]
        let c_u32 = color.to_bgra_u32();
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let width = self.width;

        loop {
            let idx = (y0 as usize) * width + (x0 as usize);
            self.pixels[idx] = c_u32;
            if thickness > 1 {
                if x0 + 1 < w {
                    self.pixels[idx + 1] = c_u32;
                }
                if y0 + 1 < h {
                    let next_row = idx + width;
                    self.pixels[next_row] = c_u32;
                    if x0 + 1 < w {
                        self.pixels[next_row + 1] = c_u32;
                    }
                }
            }

            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Відмальовування тексту растровим шрифтом 8x8
    pub fn draw_text(&mut self, start_x: i32, start_y: i32, text: &str, color: Color, scale: usize) {
        let scale = scale.max(1);
        let mut cur_x = start_x;
        let mut cur_y = start_y;

        for c in text.chars() {
            if c == '\n' {
                cur_y += (CHAR_HEIGHT * scale + 2) as i32;
                cur_x = start_x;
                continue;
            }

            let glyph = get_glyph(c);
            for row in 0..CHAR_HEIGHT {
                let byte = glyph[row];
                for col in 0..CHAR_WIDTH {
                    if (byte & (0x80 >> col)) != 0 {
                        let px = cur_x + (col * scale) as i32;
                        let py = cur_y + (row * scale) as i32;
                        if scale == 1 {
                            self.set_pixel(px, py, color);
                        } else {
                            for sy in 0..scale {
                                for sx in 0..scale {
                                    self.set_pixel(px + sx as i32, py + sy as i32, color);
                                }
                            }
                        }
                    }
                }
            }
            cur_x += (CHAR_WIDTH * scale) as i32;
        }
    }

    /// Відмальовування тексту з обмеженням за максимальною координатою X (обрізка)
    pub fn draw_text_clipped(
        &mut self,
        start_x: i32,
        start_y: i32,
        text: &str,
        color: Color,
        scale: usize,
        max_x: i32,
    ) {
        let scale = scale.max(1);
        let mut cur_x = start_x;
        let mut cur_y = start_y;
        let char_w = (CHAR_WIDTH * scale) as i32;

        for c in text.chars() {
            if c == '\n' {
                cur_y += (CHAR_HEIGHT * scale + 2) as i32;
                cur_x = start_x;
                continue;
            }

            if cur_x >= max_x {
                break;
            }

            let glyph = get_glyph(c);
            for row in 0..CHAR_HEIGHT {
                let byte = glyph[row];
                for col in 0..CHAR_WIDTH {
                    if (byte & (0x80 >> col)) != 0 {
                        let px = cur_x + (col * scale) as i32;
                        let py = cur_y + (row * scale) as i32;
                        if px < max_x {
                            if scale == 1 {
                                self.set_pixel(px, py, color);
                            } else {
                                for sy in 0..scale {
                                    for sx in 0..scale {
                                        if px + (sx as i32) < max_x {
                                            self.set_pixel(px + sx as i32, py + sy as i32, color);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            cur_x += char_w;
        }
    }
}
