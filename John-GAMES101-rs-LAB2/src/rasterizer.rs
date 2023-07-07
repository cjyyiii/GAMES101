use std::collections::HashMap;

use nalgebra::{Matrix4, Vector3, Vector4, Vector2};
use crate::triangle::Triangle;

#[allow(dead_code)]
pub enum Buffer {
    Color,
    Depth,
    Both,
}

#[allow(dead_code)]
pub enum Primitive {
    Line,
    Triangle,
}

#[derive(Default, Clone)]
pub struct Rasterizer {
    model: Matrix4<f64>,
    view: Matrix4<f64>,
    projection: Matrix4<f64>,
    pos_buf: HashMap<usize, Vec<Vector3<f64>>>,
    ind_buf: HashMap<usize, Vec<Vector3<usize>>>,
    col_buf: HashMap<usize, Vec<Vector3<f64>>>,

    frame_buf: Vec<Vector3<f64>>,
    depth_buf: Vec<f64>,
    color_buf: Vec<Vector3<f64>>,
    /*  You may need to uncomment here to implement the MSAA method  */
    // frame_sample: Vec<Vector3<f64>>,
    // depth_sample: Vec<f64>,

    width: u64,
    height: u64,
    next_id: usize,
}

#[derive(Clone, Copy)]
pub struct PosBufId(usize);

#[derive(Clone, Copy)]
pub struct IndBufId(usize);

#[derive(Clone, Copy)]
pub struct ColBufId(usize);

impl Rasterizer {
    pub fn new(w: u64, h: u64) -> Self {
        let mut r = Rasterizer::default();
        r.width = w;
        r.height = h;
        r.frame_buf.resize((w * h) as usize, Vector3::zeros());
        r.color_buf.resize((w * h) as usize, Vector3::zeros());
        r.depth_buf.resize((w * h) as usize, 0.0);
        // r.frame_sample.resize((w * h * 4) as usize, Vector3::zeros());
        // r.depth_sample.resize((w * h * 4) as usize, 0.0);
        r
    }

    fn get_index(&self, x: usize, y: usize) -> usize {
        ((self.height - 1 - y as u64) * self.width + x as u64) as usize
    }

    fn set_pixel(&mut self, point: &Vector3<f64>, color: &Vector3<f64>) {
        let ind = (self.height as f64 - 1.0 - point.y) * self.width as f64 + point.x;
        self.frame_buf[ind as usize] = *color;
    }

    fn set_col(&mut self, x: f64, y: f64, color: &Vector3<f64>) {
        // let ind = (self.height as f64 - 1.0 - point.y) * self.width as f64 + point.x;
        let ind = self.get_index(x as usize, y as usize);
        self.color_buf[ind as usize] = *color;
    }

    pub fn clear(&mut self, buff: Buffer) {
        match buff {
            Buffer::Color => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.color_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                // self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
            }
            Buffer::Depth => {
                self.depth_buf.fill(f64::MAX);
                // self.depth_sample.fill(f64::MAX);
            }
            Buffer::Both => {
                self.frame_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                self.color_buf.fill(Vector3::new(0.0, 0.0, 0.0));
                // self.frame_sample.fill(Vector3::new(0.0, 0.0, 0.0));
                self.depth_buf.fill(f64::MAX);
                // self.depth_sample.fill(f64::MAX);
            }
        }
    }

    pub fn set_model(&mut self, model: Matrix4<f64>) {
        self.model = model;
    }

    pub fn set_view(&mut self, view: Matrix4<f64>) {
        self.view = view;
    }

    pub fn set_projection(&mut self, projection: Matrix4<f64>) {
        self.projection = projection;
    }

    fn get_next_id(&mut self) -> usize {
        let res = self.next_id;
        self.next_id += 1;
        res
    }

    pub fn load_position(&mut self, positions: &Vec<Vector3<f64>>) -> PosBufId {
        let id = self.get_next_id();
        self.pos_buf.insert(id, positions.clone());
        PosBufId(id)
    }

    pub fn load_indices(&mut self, indices: &Vec<Vector3<usize>>) -> IndBufId {
        let id = self.get_next_id();
        self.ind_buf.insert(id, indices.clone());
        IndBufId(id)
    }

    pub fn load_colors(&mut self, colors: &Vec<Vector3<f64>>) -> ColBufId {
        let id = self.get_next_id();
        self.col_buf.insert(id, colors.clone());
        ColBufId(id)
    }

    pub fn draw(&mut self, pos_buffer: PosBufId, ind_buffer: IndBufId, col_buffer: ColBufId, _typ: Primitive) {
        let buf = &self.clone().pos_buf[&pos_buffer.0];
        let ind: &Vec<Vector3<usize>> = &self.clone().ind_buf[&ind_buffer.0];
        let col = &self.clone().col_buf[&col_buffer.0];

        let f1 = (50.0 - 0.1) / 2.0;
        let f2 = (50.0 + 0.1) / 2.0;

        let mvp = self.projection * self.view * self.model;

        for i in ind {
            let mut t = Triangle::new();
            let mut v =
                vec![mvp * to_vec4(buf[i[0]], Some(1.0)), // homogeneous coordinates
                     mvp * to_vec4(buf[i[1]], Some(1.0)), 
                     mvp * to_vec4(buf[i[2]], Some(1.0))];
    
            for vec in v.iter_mut() {
                *vec = *vec / vec.w;
            }
            for vert in v.iter_mut() {
                vert.x = 0.5 * self.width as f64 * (vert.x + 1.0);
                vert.y = 0.5 * self.height as f64 * (vert.y + 1.0);
                vert.z = vert.z * f1 + f2;
            }
            for j in 0..3 {
                // t.set_vertex(j, Vector3::new(v[j].x, v[j].y, v[j].z));
                t.set_vertex(j, v[j].xyz());
                t.set_vertex(j, v[j].xyz());
                t.set_vertex(j, v[j].xyz());
            }
            let col_x = col[i[0]];
            let col_y = col[i[1]];
            let col_z = col[i[2]];
            t.set_color(0, col_x[0], col_x[1], col_x[2]);
            t.set_color(1, col_y[0], col_y[1], col_y[2]);
            t.set_color(2, col_z[0], col_z[1], col_z[2]);

            self.rasterize_triangle(&t);
        }
    }

    pub fn rasterize_triangle(&mut self, t: &Triangle) {
        /*  implement your code here  */
        let v = t.to_vector4();
        let mut min_x: f64 = self.width as f64;
        let mut min_y: f64 = self.height as f64;
        let mut max_x: f64 = 0.0;
        let mut max_y: f64 = 0.0;

        for i in 0..3 {
            min_x = min_x.min(v[i].x);
            min_y = min_y.min(v[i].y);
            max_x = max_x.max(v[i].x);
            max_y = max_y.max(v[i].y);
        }
        //bounding box
        let minx: usize = min_x.floor() as usize;
        let miny: usize = min_y.floor() as usize;        
        let maxx: usize = max_x.ceil() as usize;
        let maxy: usize = max_y.ceil() as usize;
        
        // MSAA1
        // for x in minx..maxx {
        //     for y in miny..maxy {
        //         let mut degree: f64 = 0.0;
        //         if inside_triangle(x as f64 + 0.25, y as f64 + 0.25, &t.v) { degree += 0.25; }//将一个像素点分为四个
        //         if inside_triangle(x as f64 + 0.25, y as f64 + 0.75, &t.v) { degree += 0.25; }
        //         if inside_triangle(x as f64 + 0.75, y as f64 + 0.25, &t.v) { degree += 0.25; }
        //         if inside_triangle(x as f64 + 0.75, y as f64 + 0.75, &t.v) { degree += 0.25; }
        //         let index = self.get_index(x, y);
        //         if degree != 0.0 {
        //             let (alpha, beta, gamma) = compute_barycentric2d(x as f64 + 0.5, y as f64 + 0.5, &t.v);
        //             let z_interpolated = (alpha * v[0].z + beta * v[1].z + gamma * v[2].z) / (alpha + beta + gamma);//计算插值深度
                    
        //             if z_interpolated < self.depth_buf[index] {//与deep_buffer比较
        //                 let pixel: Vector3<f64> = Vector3::<f64>::new(x as f64, y as f64, z_interpolated);
        //                 self.depth_buf[index] = z_interpolated;
        //                 self.set_pixel(&pixel, &(t.get_color() * degree + (1.0 - degree) * self.frame_buf[index]));
        //                 if degree < 1.0 {
        //                     self.depth_buf[index] = std::f64::INFINITY;
        //                 }
        //             }
        //         }
        //     }
        // }

        //MSAA2
        // for x in minx..maxx {
        //     for y in miny..maxy {
        //         // let mut degree: f64 = 0.0;
        //         let index = self.get_index(x, y);
        //         if inside_triangle(x as f64 + 0.25, y as f64 + 0.25, &t.v) { 
        //             // degree += 0.25; 
        //             let (alpha, beta, gamma) = compute_barycentric2d(x as f64 + 0.25, y as f64 + 0.25, &t.v);
        //             let z_interpolated = (alpha * v[0].z + beta * v[1].z + gamma * v[2].z) / (alpha + beta + gamma);//计算插值深度
        //             if z_interpolated < self.depth_sample[index * 4] {
        //                 self.depth_sample[index * 4] = z_interpolated;
        //                 self.frame_sample[index * 4] = t.get_color() / 4.0;
        //                 self.depth_buf[index] = self.depth_buf[index].min(z_interpolated);
        //             }
        //         }//将一个像素点分为四个
        //         if inside_triangle(x as f64 + 0.25, y as f64 + 0.75, &t.v) { 
        //             // degree += 0.25; 
        //             let (alpha, beta, gamma) = compute_barycentric2d(x as f64 + 0.25, y as f64 + 0.75, &t.v);
        //             let z_interpolated = (alpha * v[0].z + beta * v[1].z + gamma * v[2].z) / (alpha + beta + gamma);//计算插值深度
        //             if z_interpolated < self.depth_sample[index * 4 + 1] {
        //                 self.depth_sample[index * 4 + 1] = z_interpolated;
        //                 self.frame_sample[index * 4 + 1] = t.get_color() / 4.0;
        //                 self.depth_buf[index] = self.depth_buf[index].min(z_interpolated);
        //             }
        //         }
        //         if inside_triangle(x as f64 + 0.75, y as f64 + 0.25, &t.v) { 
        //             // degree += 0.25; 
        //             let (alpha, beta, gamma) = compute_barycentric2d(x as f64 + 0.75, y as f64 + 0.25, &t.v);
        //             let z_interpolated = (alpha * v[0].z + beta * v[1].z + gamma * v[2].z) / (alpha + beta + gamma);//计算插值深度
        //             if z_interpolated < self.depth_sample[index * 4 + 2] {
        //                 self.depth_sample[index * 4 + 2] = z_interpolated;
        //                 self.frame_sample[index * 4 + 2] = t.get_color() / 4.0;
        //                 self.depth_buf[index] = self.depth_buf[index].min(z_interpolated);
        //             }
        //         }
        //         if inside_triangle(x as f64 + 0.75, y as f64 + 0.75, &t.v) { 
        //             // degree += 0.25; 
        //             let (alpha, beta, gamma) = compute_barycentric2d(x as f64 + 0.75, y as f64 + 0.75, &t.v);
        //             let z_interpolated = (alpha * v[0].z + beta * v[1].z + gamma * v[2].z) / (alpha + beta + gamma);//计算插值深度
        //             if z_interpolated < self.depth_sample[index * 4 + 3] {
        //                 self.depth_sample[index * 4 + 3] = z_interpolated;
        //                 self.frame_sample[index * 4 + 3] = t.get_color() / 4.0;
        //                 self.depth_buf[index] = self.depth_buf[index].min(z_interpolated);
        //             }
        //         }
        //         self.set_pixel(&Vector3::new(x as f64, y as f64, self.depth_buf[index]), &(self.frame_sample[index * 4] + self.frame_sample[index * 4 + 1] + self.frame_sample[index * 4 + 2] + self.frame_sample[index * 4 + 3]));
        //     }
        // }

        
        for x in minx..maxx {
            for y in miny..maxy {
                if inside_triangle(x as f64 + 0.5, y as f64 + 0.5, &t.v) { 
                let index = self.get_index(x, y);
                    let (alpha, beta, gamma) = compute_barycentric2d(x as f64 + 0.5, y as f64 + 0.5, &t.v);
                    let z_interpolated = (alpha * v[0].z + beta * v[1].z + gamma * v[2].z) / (alpha + beta + gamma);//计算插值深度
                    
                    if z_interpolated < self.depth_buf[index] {//与deep_buffer比较
                        // let pixel: Vector3<f64> = Vector3::<f64>::new(x as f64, y as f64, z_interpolated);
                        self.depth_buf[index] = z_interpolated;
                        self.set_col(x as f64, y as f64, &t.get_color());

                    }
                }
            }
        }

        for x in minx..maxx {
            for y in miny..maxy {
                let index_m = self.get_index(x, y);
                let index_w = self.get_index(x - 1, y);
                let index_n = self.get_index(x, y + 1);
                let index_e = self.get_index(x + 1, y);
                let index_s = self.get_index(x, y - 1);
                let index_nw = self.get_index(x - 1, y + 1);
                let index_sw = self.get_index(x - 1, y - 1);
                let index_ne = self.get_index(x + 1, y + 1);
                let index_se =  self.get_index(x + 1, y - 1);
                let m = self.color_buf[index_m];
                let w = self.color_buf[index_w];
                let n = self.color_buf[index_n];
                let e = self.color_buf[index_e];
                let s = self.color_buf[index_s];
                let nw = self.color_buf[index_nw];
                let sw = self.color_buf[index_sw];
                let ne = self.color_buf[index_ne];
                let se = self.color_buf[index_se];
                let color = 1.0 / 5.0 * m + 2.0 / 15.0 * (w + n + e + s) + 1.0 / 15.0 * (nw + sw + ne + se);
                self.set_pixel(&Vector3::new(x as f64, y as f64, 0.0), &color);
            }
        }

    }

    pub fn frame_buffer(&self) -> &Vec<Vector3<f64>> {
        &self.frame_buf
    }

}

// fn clamp(x: f64, min: f64, max: f64) -> f64 {
//     if x < min {
//         return min;
//     }
//     if x > max {
//         return max;
//     }
//     x
// }

fn to_vec4(v3: Vector3<f64>, w: Option<f64>) -> Vector4<f64> {
    Vector4::new(v3.x, v3.y, v3.z, w.unwrap_or(1.0))
}

fn inside_triangle(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> bool {
    /*  implement your code here  */
    let side1: Vector2<f64> = Vector2::<f64>::new(v[1].x - v[0].x, v[1].y - v[0].y);
    let side2: Vector2<f64> = Vector2::<f64>::new(v[2].x - v[1].x, v[2].y - v[1].y);
    let side3: Vector2<f64> = Vector2::<f64>::new(v[0].x - v[2].x, v[0].y - v[2].y);

    let v1: Vector2<f64> = Vector2::<f64>::new(x - v[0].x, y - v[0].y);
    let v2: Vector2<f64> = Vector2::<f64>::new(x - v[1].x, y - v[1].y);
    let v3: Vector2<f64> = Vector2::<f64>::new(x - v[2].x, y - v[2].y);
    
    let z1: f64 = side1.x * v1.y - side1.y * v1.x;//叉乘
    let z2: f64 = side2.x * v2.y - side2.y * v2.x;
    let z3: f64 = side3.x * v3.y - side3.y * v3.x;
    
    if (z1 >= 0.0 && z2 >= 0.0 && z3 >= 0.0) || (z1 <= 0.0 && z2 <= 0.0 && z3 <= 0.0) {
        return true;
    }
    false
}

fn compute_barycentric2d(x: f64, y: f64, v: &[Vector3<f64>; 3]) -> (f64, f64, f64) {
    let c1 = (x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * y + v[1].x * v[2].y - v[2].x * v[1].y)
        / (v[0].x * (v[1].y - v[2].y) + (v[2].x - v[1].x) * v[0].y + v[1].x * v[2].y - v[2].x * v[1].y);
    let c2 = (x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * y + v[2].x * v[0].y - v[0].x * v[2].y)
        / (v[1].x * (v[2].y - v[0].y) + (v[0].x - v[2].x) * v[1].y + v[2].x * v[0].y - v[0].x * v[2].y);
    let c3 = (x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * y + v[0].x * v[1].y - v[1].x * v[0].y)
        / (v[2].x * (v[0].y - v[1].y) + (v[1].x - v[0].x) * v[2].y + v[0].x * v[1].y - v[1].x * v[0].y);
    (c1, c2, c3)
}