#![allow(dead_code)] // TODO: remove this
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, OMatrix, OVector, RealField};
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::{Standard, StandardNormal};
use rustc_hash::FxHashMap;

use crate::localization::bayesian_filter::{BayesianFilter, BayesianFilterKnownCorrespondences};
use crate::models::measurement::MeasurementModel;
use crate::models::motion::MotionModel;
use crate::utils::mvn::MultiVariateNormal;
use crate::utils::state::GaussianState;

pub enum ResamplingScheme {
    IID,
    Stratified,
    Systematic,
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ParticleFilter<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    r: OMatrix<T, S, S>,
    q: OMatrix<T, Z, Z>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    pub particules: Vec<OVector<T, S>>,
    resampling_scheme: ResamplingScheme,
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> ParticleFilter<T, S, Z, U>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    pub fn new(
        r: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        initial_state: GaussianState<T, S>,
        num_particules: usize,
        resampling_scheme: ResamplingScheme,
    ) -> ParticleFilter<T, S, Z, U> {
        let mvn = MultiVariateNormal::new(&initial_state.x, &r).unwrap();
        let mut particules = Vec::with_capacity(num_particules);
        for _ in 0..num_particules {
            particules.push(mvn.sample());
        }

        ParticleFilter {
            r,
            q,
            measurement_model,
            motion_model,
            particules,
            resampling_scheme,
        }
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> BayesianFilter<T, S, Z, U>
    for ParticleFilter<T, S, Z, U>
where
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
    Standard: Distribution<T>,
    StandardNormal: Distribution<T>,
{
    fn update_estimate(&mut self, u: &OVector<T, U>, z: &OVector<T, Z>, dt: T) {
        let shape = self.particules[0].shape_generic();
        let mvn =
            MultiVariateNormal::new(&OMatrix::zeros_generic(shape.0, shape.1), &self.r).unwrap();

        self.particules = self
            .particules
            .iter()
            .map(|p| self.motion_model.prediction(p, u, dt) + mvn.sample())
            .collect();

        let mut weights = vec![T::one(); self.particules.len()];
        let shape = z.shape_generic();
        let mvn =
            MultiVariateNormal::new(&OMatrix::zeros_generic(shape.0, shape.1), &self.q).unwrap();

        for (i, particule) in self.particules.iter().enumerate() {
            let z_pred = self.measurement_model.prediction(particule, None);
            let error = z - z_pred;
            let pdf = mvn.pdf(&error);
            weights[i] *= pdf;
        }

        self.particules = match self.resampling_scheme {
            ResamplingScheme::IID => resampling_sort(&self.particules, &weights),
            ResamplingScheme::Stratified => resampling_stratified(&self.particules, &weights),
            ResamplingScheme::Systematic => resampling_systematic(&self.particules, &weights),
        };
    }

    fn gaussian_estimate(&self) -> GaussianState<T, S> {
        gaussian_estimate(&self.particules)
    }
}

/// S : State Size, Z: Observation Size, U: Input Size
pub struct ParticleFilterKnownCorrespondences<T: RealField, S: Dim, Z: Dim, U: Dim>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z>,
{
    q: OMatrix<T, Z, Z>,
    landmarks: FxHashMap<u32, OVector<T, S>>,
    measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
    motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
    pub particules: Vec<OVector<T, S>>,
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> ParticleFilterKnownCorrespondences<T, S, Z, U>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
    DefaultAllocator:
        Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Z, Z> + Allocator<T, Const<1>, S>,
{
    pub fn new(
        initial_noise: OMatrix<T, S, S>,
        q: OMatrix<T, Z, Z>,
        landmarks: FxHashMap<u32, OVector<T, S>>,
        measurement_model: Box<dyn MeasurementModel<T, S, Z> + Send>,
        motion_model: Box<dyn MotionModel<T, S, Z, U> + Send>,
        initial_state: GaussianState<T, S>,
        num_particules: usize,
    ) -> ParticleFilterKnownCorrespondences<T, S, Z, U> {
        let mvn = MultiVariateNormal::new(&initial_state.x, &initial_noise).unwrap();
        let mut particules = Vec::with_capacity(num_particules);
        for _ in 0..num_particules {
            particules.push(mvn.sample());
        }

        ParticleFilterKnownCorrespondences {
            q,
            landmarks,
            measurement_model,
            motion_model,
            particules,
        }
    }
}

impl<T: RealField + Copy, S: Dim, Z: Dim, U: Dim> BayesianFilterKnownCorrespondences<T, S, Z, U>
    for ParticleFilterKnownCorrespondences<T, S, Z, U>
where
    StandardNormal: Distribution<T>,
    Standard: Distribution<T>,
    DefaultAllocator: Allocator<T, S>
        + Allocator<T, U>
        + Allocator<T, Z>
        + Allocator<T, S, S>
        + Allocator<T, Z, Z>
        + Allocator<T, Z, S>
        + Allocator<T, S, U>
        + Allocator<T, U, U>
        + Allocator<T, Const<1>, S>
        + Allocator<T, Const<1>, Z>,
{
    fn update_estimate(
        &mut self,
        control: Option<OVector<T, U>>,
        measurements: Option<Vec<(u32, OVector<T, Z>)>>,
        dt: T,
    ) {
        if let Some(u) = control {
            self.particules = self
                .particules
                .iter()
                .map(|p| self.motion_model.sample(p, &u, dt))
                .collect();
        }

        if let Some(measurements) = measurements {
            let mut weights = vec![T::one(); self.particules.len()];
            let shape = measurements[0].1.shape_generic();
            let mvn = MultiVariateNormal::new(&OMatrix::zeros_generic(shape.0, shape.1), &self.q)
                .unwrap();

            for (id, z) in measurements
                .iter()
                .filter(|(id, _)| self.landmarks.contains_key(id))
            {
                let landmark = self.landmarks.get(id);
                for (i, particule) in self.particules.iter().enumerate() {
                    let z_pred = self.measurement_model.prediction(particule, landmark);
                    let error = z - z_pred;
                    let pdf = mvn.pdf(&error);
                    weights[i] *= pdf;
                }
            }
            self.particules = resampling(&self.particules, &weights);
            // self.particules = resampling_sort(&self.particules, weights);
        }
    }

    fn gaussian_estimate(&self) -> GaussianState<T, S> {
        gaussian_estimate(&self.particules)
    }
}

fn gaussian_estimate<T: RealField + Copy, S: Dim>(
    particules: &[OVector<T, S>],
) -> GaussianState<T, S>
where
    DefaultAllocator: Allocator<T, S> + Allocator<T, S, S> + Allocator<T, Const<1>, S>,
{
    let shape = particules[0].shape_generic();
    let x = particules
        .iter()
        .fold(OMatrix::zeros_generic(shape.0, shape.1), |a, b| a + b)
        / T::from_usize(particules.len()).unwrap();
    let cov = particules
        .iter()
        .map(|p| p - &x)
        .map(|dx| &dx * dx.transpose())
        .fold(OMatrix::zeros_generic(shape.0, shape.0), |a, b| a + b)
        / T::from_usize(particules.len()).unwrap();
    GaussianState { x, cov }
}

fn resampling<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let cum_weight: Vec<T> = weights
        .iter()
        .scan(T::zero(), |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    let weight_tot = *cum_weight.last().unwrap();

    // sampling
    let mut rng = rand::thread_rng();
    (0..particules.len())
        .map(|_| {
            let rng_nb = rng.gen::<T>() * weight_tot;
            for (i, p) in particules.iter().enumerate() {
                if (&cum_weight)[i] > rng_nb {
                    return p.clone();
                }
            }
            unreachable!()
        })
        .collect()
}

fn resampling_sort<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
    let mut rng = rand::thread_rng();
    let mut draws: Vec<T> = (0..particules.len())
        .map(|_| rng.gen::<T>() * total_weight)
        .collect();
    resample(&mut draws, total_weight, particules, weights)
}

fn resampling_stratified<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
    let mut rng = rand::thread_rng();
    let mut draws: Vec<T> = (0..particules.len())
        .map(|i| {
            (T::from_usize(i).unwrap() + rng.gen::<T>()) / T::from_usize(particules.len()).unwrap()
                * total_weight
        })
        .collect();
    resample(&mut draws, total_weight, particules, weights)
}

fn resampling_systematic<T: RealField + Copy, S: Dim>(
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    let total_weight: T = weights.iter().fold(T::zero(), |a, b| a + *b);
    let mut rng = rand::thread_rng();
    let draw = rng.gen::<T>();
    let mut draws: Vec<T> = (0..particules.len())
        .map(|i| {
            (T::from_usize(i).unwrap() + draw) / T::from_usize(particules.len()).unwrap()
                * total_weight
        })
        .collect();
    resample(&mut draws, total_weight, particules, weights)
}

fn resample<T: RealField + Copy, S: Dim>(
    draws: &mut [T],
    total_weight: T,
    particules: &Vec<OVector<T, S>>,
    weights: &[T],
) -> Vec<OVector<T, S>>
where
    DefaultAllocator: Allocator<T, S>,
    Standard: Distribution<T>,
{
    draws.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut index = 0;
    let mut cum_weight = draws[0];
    (0..particules.len())
        .map(|i| {
            while cum_weight < draws[i] {
                if index == particules.len() - 1 {
                    // weird precision edge case
                    cum_weight = total_weight;
                    break;
                } else {
                    cum_weight += weights[index];
                    index += 1;
                }
            }
            particules[index].clone()
        })
        .collect()
}
