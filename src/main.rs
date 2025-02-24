use burn::{
    module::Module,
    nn::{Linear, MSELoss},
    tensor::{backend::AutodiffBackend, Data, Tensor},
    train::{LearnerBuilder, RegressionOutput},
};
use burn_ndarray::NdArrayBackend;
use rand::Rng;
use textplots::{Chart, Plot, Shape};

type Backend = NdArrayBackend<f32>;

#[derive(Module, Debug)]
struct LinearRegressionModel<B: AutodiffBackend> {
    linear: Linear<B>,
}

impl<B: AutodiffBackend> LinearRegressionModel<B> {
    fn new() -> Self {
        Self {
            linear: Linear::new(1, 1),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

fn generate_data(num_samples: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();

    for _ in 0..num_samples {
        let x: f32 = rng.gen_range(-10.0..10.0);
        let y: f32 = 2.0 * x + 1.0 + rng.gen_range(-1.0..1.0); // Adding noise
        x_vals.push(x);
        y_vals.push(y);
    }

    (x_vals, y_vals)
}

fn main() {
    let (x_vals, y_vals) = generate_data(100);

    let x_tensor = Tensor::<Backend, 2>::from_data(Data::from_vec(x_vals.clone(), [100, 1]));
    let y_tensor = Tensor::<Backend, 2>::from_data(Data::from_vec(y_vals.clone(), [100, 1]));

    let model = LinearRegressionModel::<Backend>::new();
    let criterion = MSELoss::new();
    let optimizer = burn::optim::Adam::new();

    let learner = LearnerBuilder::new(&model)
        .loss(criterion)
        .metric_fn(RegressionOutput::mse)
        .optimizer(optimizer)
        .build();

    learner.fit(x_tensor.clone(), y_tensor.clone(), 10);

    // Testing the model
    let test_x: Vec<f32> = (-10..10).map(|x| x as f32).collect();
    let test_x_tensor =
        Tensor::<Backend, 2>::from_data(Data::from_vec(test_x.clone(), [test_x.len(), 1]));
    let test_y_pred = model.forward(test_x_tensor).into_data().value;

    // Plot results
    println!("Plotting results...");
    Chart::new(180, 60, -10.0, 10.0)
        .lineplot(&Shape::Continuous(|x| 2.0 * x + 1.0)) // True function
        .points(&test_x.iter().zip(test_y_pred.iter()).map(|(x, y)| (*x, *y)).collect::<Vec<_>>())
        .display();
}

