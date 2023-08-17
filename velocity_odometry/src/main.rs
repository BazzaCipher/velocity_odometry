use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, Matrix2, Matrix3, Matrix3x2, Matrix3x6,
    Matrix6, OMatrix, OVector, RealField, Vector2, Vector3, Vector6,
};
use nannou::prelude::*;
use rand::Rng;
use robotics::{
    localization::{BayesianFilterKnownCorrespondences, ParticleFilterKnownCorrespondences},
    models::{measurement::RangeBearingMeasurementModel, motion::MotionModel},
    utils::state::GaussianState,
};
use rustc_hash::FxHashMap;
use std::time::Duration;

// Robot pose is a tuple of x, y, bearing
type RobotState<T> = Vector3<T>;
// Control data of the current and previous pose for some small timestep
type OdometryControlData<'a, T> = Vector6<T>;
type DummyBeamModel = RangeBearingMeasurementModel;
type PFKnown<T> = ParticleFilterKnownCorrespondences<T, Const<3>, Const<2>, Const<6>>;
type BeamMeasurementData<T> = Vector2<T>;
type CorrespondenceMap<T, V> = FxHashMap<T, V>;

struct OdometryModel {
    // these variables are the noise hyperparameters (strictly not named as hyperparameters)
    a: [f64; 6],
}

impl OdometryModel {
    fn new(a: [f64; 6]) -> Self {
        OdometryModel { a }
    }
}

impl MotionModel<f64, Const<3>, Const<2>, Const<6>> for OdometryModel
// The left is the robot pose and that of landmarks (usually ignore the third)
// The middle is the number of data points received from sensors (range, bearing for beams)
// The right is the control model number of variables
{
    fn prediction(
        &self,
        x: &RobotState<f64>,
        u: &OdometryControlData<f64>,
        dt: f64,
    ) -> RobotState<f64> {
        // Odometry model prediction; dead reckoning
        Vector3::from_vec(vec![u.x, u.y, u.z])
    }

    fn jacobian_wrt_state(
        &self,
        x: &RobotState<f64>,
        u: &OdometryControlData<f64>,
        dt: f64,
    ) -> Matrix3<f64> {
        todo!()
    }

    fn jacobian_wrt_input(
        &self,
        x: &RobotState<f64>,
        u: &OdometryControlData<f64>,
        dt: f64,
    ) -> Matrix3x6<f64> {
        todo!()
    }

    fn cov_noise_control_space(&self, u: &OdometryControlData<f64>) -> Matrix6<f64> {
        todo!()
    }

    fn sample(
        &self,
        x: &RobotState<f64>,
        u: &OdometryControlData<f64>,
        dt: f64,
    ) -> RobotState<f64> {
        // Samples with distributions according to some parameters
        let [a1, a2, a3, a4] = [1., 1., 1., 1.] else { unreachable!() };
        let [rx, ry, rz] = x.as_slice() else { unreachable!() };
        let [x0, y0, z0, x1, y1, z1] = u.as_slice() else { unreachable!() };

        let r1 = f64::atan2(y1 - y0, x1 - x0) - z0;
        let tr = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
        let r2 = z1 - z0 - r1;

        let pr1 = r1 - sample(a1 * r1 * r1 + a2 * tr * tr + 0.01);
        let ptr = tr - sample(a3 * tr * tr + a4 * (r1 * r1 + r2 * r2 + 0.01));
        let pr2 = r2 - sample(a1 * r2 * r2 + a2 * tr * tr + 0.01);

        RobotState::new(
            rx + ptr * (rz + pr1).cos(),
            ry + ptr * (rz + pr1).sin(),
            rz + pr1 + pr2,
        )
    }
}

struct Model {
    _window: window::Id,
    filter: Box<PFKnown<f64>>,
}

fn model(app: &App) -> Model {
    let _window = app.new_window().view(view).build().unwrap();

    // So we create a particle model, and with random data try to test the model
    let dummymodel = DummyBeamModel::new();

    let mut map = CorrespondenceMap::<u32, RobotState<f64>>::default();
    map.insert(1, RobotState::from_element(30.));
    map.insert(2, RobotState::from_vec(vec![5., 15., 40.]));
    map.insert(3, RobotState::from_element(-15.));

    println!("Landmarks: {:?}", map);
    println!("Control: 40, 0. rad");

    let mut pf: PFKnown<f64> = PFKnown::<f64>::new(
        30. * random_positive_semidefinite3(), // Position covariance
        30. * random_positive_semidefinite2(), // Sensor covariance
        map,                                   // Landmark correspondences
        DummyBeamModel::new(),                 // Measurement model
        Box::new(OdometryModel::new([1., 1., 1., 1., 1., 1.])), // Motion model
        // OVector::<f64, Const<2>>::from_element(1.),    // Initial pose
        GaussianState {
            x: OVector::<f64, Const<3>>::from_element(0.),
            cov: 10. * random_positive_semidefinite3(),
        }, // Initial position
        100, // Number of particles
    );

    Model {
        _window,
        filter: Box::new(pf),
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    // Recall that we assume that the robot does not move and is located
    model.filter.update_estimate(
        Some(OVector::<f64, Const<6>>::from_element(0.)),
        Some(vec![(1, BeamMeasurementData::from_vec(vec![40., 0.8]))]), // Hypothesis 1: Compares the actual position to the expected position, Hypothesis 2: Compares the range and bearing of the real and expected values
        0.1, // Doesn't make a fucking difference
    );
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(WHITE);
    for particle in &model.filter.particules {
        draw.ellipse().color(STEELBLUE).w_h(20., 20.).x_y(particle.x as f32, particle.y as f32);
    }
    let estimate = model.filter.gaussian_estimate().x;
    draw.ellipse()
        .color(PINK)
        .w_h(20., 20.)
        .x_y(estimate.x as f32, estimate.y as f32);
    draw.to_frame(app, &frame).unwrap();
}

fn sample(var: f64) -> f64 {
    assert!(var > 0.0, "Variance of sampling must be positive");
    sample_normal(var)
}

fn sample_normal(var: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let stddev = var.sqrt();
    0.5 * (0..12).fold(0., |a, _| a + rng.gen_range(-stddev..stddev))
}

fn main() {
    nannou::app(model)
        .loop_mode(LoopMode::Rate { update_interval: Duration::from_secs(5000) })
        .update(update)
        .run();
}

#[cfg(test)]
#[test]
fn test_hone_on_position() {
    // So we create a particle model, and with random data try to test the model
    let dummymodel = DummyBeamModel::new();

    let mut map = CorrespondenceMap::<u32, RobotState<f64>>::default();
    map.insert(1, RobotState::from_element(30.));
    map.insert(2, RobotState::from_vec(vec![5., 15., 40.]));
    map.insert(3, RobotState::from_element(-15.));

    println!("Landmarks: {:?}", map);
    println!("Control: 40, 0. rad");

    let mut pf: PFKnown<f64> = PFKnown::<f64>::new(
        30. * random_positive_semidefinite3(), // Position covariance
        30. * random_positive_semidefinite2(), // Sensor covariance
        map,                                   // Landmark correspondences
        DummyBeamModel::new(),                 // Measurement model
        Box::new(OdometryModel::new([1., 1., 1., 1., 1., 1.])), // Motion model
        GaussianState {
            x: OVector::<f64, Const<3>>::from_element(100.),
            cov: 10. * random_positive_semidefinite3(),
        }, // Initial position
        1, // Number of particles
    );

    // First time step
    pf.update_estimate(
        Some(OVector::<f64, Const<6>>::from_element(0.)),
        Some(vec![(1, BeamMeasurementData::from_vec(vec![40., 0.8]))]), // Hypothesis 1: Compares the actual position to the expected position, Hypothesis 2: Compares the range and bearing of the real and expected values
        0.1, // Doesn't make a fucking difference
    );

    // Best guess
    println!("{:?}", pf.gaussian_estimate());

    // Second time step
    pf.update_estimate(
        Some(OVector::<f64, Const<6>>::from_element(0.)),
        Some(vec![(1, BeamMeasurementData::from_vec(vec![40., 0.8]))]), // Hypothesis 1: Compares the actual position to the expected position, Hypothesis 2: Compares the range and bearing of the real and expected values
        0.1, // Doesn't make a fucking difference
    );

    // Best guess
    println!("{:?}", pf.gaussian_estimate());

    // Third time step
    pf.update_estimate(
        Some(OVector::<f64, Const<6>>::from_element(0.)),
        Some(vec![(1, BeamMeasurementData::from_vec(vec![40., 0.8]))]), // Hypothesis 1: Compares the actual position to the expected position, Hypothesis 2: Compares the range and bearing of the real and expected values
        0.1, // Doesn't make a fucking difference
    );

    // Best guess
    println!("Best guess: {:?}", pf.gaussian_estimate());
}

#[test]
fn test_random_positive_hermitian() {
    let c = random_positive_semidefinite2();
    let b = random_positive_semidefinite3();
    println!("{:?}\r\n{:?}", c, b);
}

fn random_positive_semidefinite3() -> OMatrix<f64, Const<3>, Const<3>> {
    let m = OMatrix::<f64, Const<3>, Const<3>>::new_random();
    m * m.transpose()
}
fn random_positive_semidefinite2() -> OMatrix<f64, Const<2>, Const<2>> {
    let m = OMatrix::<f64, Const<2>, Const<2>>::new_random();
    m * m.transpose()
}
