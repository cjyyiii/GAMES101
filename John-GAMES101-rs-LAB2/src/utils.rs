use std::os::raw::c_void;
use nalgebra::{Matrix4, Vector3};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color};

pub(crate) fn get_view_matrix(eye_pos: Vector3<f64>) -> Matrix4<f64> {
    // let mut view: Matrix4<f64> = Matrix4::identity();
    /*  implement what you've done in LAB1  */
    let view: Matrix4<f64> = Matrix4::<f64>::new(
        1.0, 0.0, 0.0, -eye_pos.x,
        0.0, 1.0, 0.0, -eye_pos.y,
        0.0, 0.0, 1.0, -eye_pos.z,
        0.0, 0.0, 0.0, 1.0,
    );
    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64) -> Matrix4<f64> {
    // let mut model: Matrix4<f64> = Matrix4::identity();
    /*  implement what you've done in LAB1  */
    let radian: f64 = (rotation_angle / 180.0) * std::f64::consts::PI;
    let model: Matrix4<f64> = Matrix4::<f64>::new(
        radian.cos(), -radian.sin(), 0.0, 0.0,
        radian.sin(), radian.cos(), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    model
}

pub(crate) fn get_projection_matrix(eye_fov: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Matrix4<f64> {
    // let mut projection: Matrix4<f64> = Matrix4::identity();
    /*  implement what you've done in LAB1  */
    let projection: Matrix4<f64> = Matrix4::<f64>::new(
        z_near, 0.0, 0.0, 0.0,
        0.0, z_near, 0.0, 0.0,
        0.0, 0.0, z_near + z_far, -z_near * z_far,
        0.0, 0.0, 1.0, 0.0, 
    );
    let radian = eye_fov / 2.0 * std::f64::consts::PI / 180.0;
    let t = radian.tan() * -z_near;
    let r = t * aspect_ratio;
    let b = -t;
    let l = -r;
    let ortho1: Matrix4<f64> = Matrix4::<f64>::new(
        2.0 / (r - l), 0.0, 0.0, 0.0,
        0.0, 2.0 / (t - b), 0.0, 0.0, 
        0.0, 0.0, 2.0 / (z_near - z_far), 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let ortho2: Matrix4<f64> = Matrix4::<f64>::new(
        1.0, 0.0, 0.0, -(r + l) / 2.0,
        0.0, 1.0, 0.0, -(t + b) / 2.0,
        0.0, 0.0, 1.0, -(z_near + z_far) / 2.0,
        0.0, 0.0, 0.0, 1.0,
    );
    let ortho: Matrix4<f64> =  ortho1 * ortho2;
    ortho * projection
}


pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<Vector3<f64>>) -> opencv::core::Mat {
    let mut image = unsafe {
        Mat::new_rows_cols_with_data(
            700, 700,
            opencv::core::CV_64FC3,
            frame_buffer.as_ptr() as *mut c_void,
            opencv::core::Mat_AUTO_STEP,
        ).unwrap()
    };
    let mut img = Mat::copy(&image).unwrap();
    image.convert_to(&mut img, opencv::core::CV_8UC3, 1.0, 1.0).expect("panic message");
    cvt_color(&img, &mut image, COLOR_RGB2BGR, 0).unwrap();
    image
}