use std::os::raw::c_void;
use nalgebra::{Matrix4, Vector3, Matrix3};
use opencv::core::{Mat, MatTraitConst};
use opencv::imgproc::{COLOR_RGB2BGR, cvt_color};

pub type V3d = Vector3<f64>;

pub(crate) fn get_view_matrix(eye_pos: V3d) -> Matrix4<f64> {
    // let mut view: Matrix4<f64> = Matrix4::identity();
    /*  implement your code here  */
    let view: Matrix4<f64> = Matrix4::<f64>::new(
        1.0, 0.0, 0.0, -eye_pos.x,
        0.0, 1.0, 0.0, -eye_pos.y,
        0.0, 0.0, 1.0, -eye_pos.z,
        0.0, 0.0, 0.0, 1.0,
    );
    view
}

pub(crate) fn get_model_matrix(rotation_angle: f64) -> Matrix4<f64> {
    // let model: Matrix4<f64> = Matrix4::identity();
    /*  implement your code here  */
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
    /*  implement your code here  */
    let projection: Matrix4<f64> = Matrix4::<f64>::new(
        z_near, 0.0, 0.0, 0.0,
        0.0, z_near, 0.0, 0.0,
        0.0, 0.0, z_near + z_far, -z_near * z_far,
        0.0, 0.0, 1.0, 0.0, 
    );
    let radian = eye_fov / 2.0 * std::f64::consts::PI / 180.0;
    let r = radian.tan() * z_near;
    let b = r * aspect_ratio;
    let t = -b;
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

pub(crate) fn get_rotation(axis: V3d, angle: f64) -> Matrix4<f64> {
    let radian = angle * std::f64::consts::PI / 180.0;
    let mul: Matrix3<f64> = Matrix3::new(
        0.0, -axis[2], axis[1],
        axis[2], 0.0, -axis[0],
        -axis[1], axis[0], 0.0,
    );
    let temp: Matrix3<f64> = Matrix3::identity();
    let _temp: Matrix3<f64> = radian.cos() * temp + (1.0 - radian.cos()) * axis * axis.adjoint() + radian.sin() * mul;
    let model: Matrix4<f64> = Matrix4::new(
        _temp.m11, _temp.m12, _temp.m13, 0.0,
        _temp.m21, _temp.m22, _temp.m23, 0.0,
        _temp.m31, _temp.m32, _temp.m33, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );
    model
}

pub(crate) fn frame_buffer2cv_mat(frame_buffer: &Vec<V3d>) -> opencv::core::Mat {
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