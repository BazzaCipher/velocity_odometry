use criterion::{criterion_group, criterion_main, Criterion};

use nalgebra::{Const, Matrix4, Vector2, Vector4};
extern crate robotics;
use robotics::localization::{BayesianFilter, ExtendedKalmanFilter, UnscentedKalmanFilter};
use robotics::models::measurement::SimpleProblemMeasurementModel;
use robotics::models::motion::SimpleProblemMotionModel;
use robotics::utils::deg2rad;
use robotics::utils::state::GaussianState;

fn ekf(b: &mut Criterion) {
    // setup ekf
    let q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
    let r = nalgebra::Matrix2::identity();
    let motion_model = SimpleProblemMotionModel::new();
    let measurement_model = SimpleProblemMeasurementModel::new();
    let initial_state = GaussianState {
        x: Vector4::<f64>::new(0., 0., 0., 0.),
        cov: Matrix4::<f64>::identity(),
    };
    let mut ekf = ExtendedKalmanFilter::<f64, Const<4>, Const<2>, Const<2>>::new(
        q,
        r,
        measurement_model,
        motion_model,
        initial_state,
    );

    let dt = 0.1;
    let u: Vector2<f64> = Default::default();
    let z: Vector2<f64> = Default::default();

    b.bench_function("ekf", |b| b.iter(|| ekf.update_estimate(&u, &z, dt)));
}

fn ukf(b: &mut Criterion) {
    // setup ukf
    let dt = 0.1;
    let q = Matrix4::<f64>::from_diagonal(&Vector4::new(0.1, 0.1, deg2rad(1.0), 1.0));
    let r = nalgebra::Matrix2::identity(); //Observation x,y position covariance
    let initial_state = GaussianState {
        x: Vector4::<f64>::new(0., 0., 0., 0.),
        cov: Matrix4::<f64>::identity(),
    };
    let mut ukf = UnscentedKalmanFilter::<f64, Const<4>, Const<2>, Const<2>>::new(
        q,
        r,
        Box::new(SimpleProblemMeasurementModel {}),
        Box::new(SimpleProblemMotionModel {}),
        0.1,
        2.0,
        0.0,
        initial_state,
    );

    let u: Vector2<f64> = Default::default();
    let z: Vector2<f64> = Default::default();

    b.bench_function("ukf", |b| b.iter(|| ukf.update_estimate(&u, &z, dt)));
}

criterion_group!(benches, ekf, ukf);
criterion_main!(benches);
