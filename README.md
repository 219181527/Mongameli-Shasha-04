# Linear Regression Model in Rust

## Introduction

This project aims to implement a **linear regression model** in **Rust** using the **Burn** library. The goal is to train a model that learns the relationship `y = 2x + 1` using synthetic data with noise. This helps in understanding how machine learning can be implemented in Rust.

Despite following the planned approach, I encountered several issues while running the code, preventing me from achieving full execution. This README documents my approach, the errors I faced, and what I learned from the process.

---

## Setup Instructions

### Prerequisites

Ensure you have the following installed before attempting to run the project:

1. **Rust** – Install Rust using (https://www.rust-lang.org/tools/install).
2. **Cargo** – Rust's package manager (comes with Rust).
3. **Git** – Install Git from (https://git-scm.com/downloads).
4. **VS Code or Rust Rover** 

### Steps to Set Up the Project

1. **Create the Repository**  
   git init
   git remote add origin https://github.com/219181527/Mongameli-Shasha-04.git
   git add .
   git commit -m "First Commit"
   git push -u origin master

   
3. ** Add the required Dependencies to Cargo.toml **
   [ dependencie s ]
burn = { version = ” 0.16.0 ” , features = [ ”wgpu” ,” train” ] }
burn-ndarray = ”0.16.0”
rand = ”0.9.0”
rgb = ”0.8.50”
textplots = ”0.8.6”

4. ** Install Dependencies **
   cargo build

5. ** Run the Code **
   cargo run

** Approach **
1. Generating Synthetic Data
The first step was to create a dataset of (x, y) pairs, where y = 2x + 1. To simulate real-world scenarios, I added random noise to the data.

2. Defining the Model
I attempted to use the Burn library to define a simple linear regression model. The model was designed with:

A single linear layer
Mean Squared Error (MSE) loss function
Adam optimizer

3. Training the Model
The plan was to use gradient descent to train the model, monitor its loss, and adjust the weights accordingly.

4. Evaluating the Model
Once trained, I intended to test the model on unseen data and visualize the results using the textplots crate.







Errors and Challenges
1. Dependency Issues with Burn
I encountered multiple unresolved imports related to the Burn library, such as:
E0432: Unresolved import: `burn_ndarray::NdArrayBackend` E0425: Cannot find function `mse_loss` in this scope


2. Type Mismatch Errors
Errors such as:
E0277: Trait `From<Vec<[f32; 1], Global>>` is not implemented for `TensorData`


3. Git Issues (git push -u origin main)
! [rejected] main -> main (fetch first)
error: failed to push some refs to 'https://github.com/219181527/Mongameli-Shasha-03.git'

** Reflection on My Learning Process **

*How Much Help I Received: *

AI Assistance: I used ChatGPT to debug the errors and understand Rust's strict type system.
Documentation: I referred to the Burn documentation and Rust’s official documentation to troubleshoot dependency issues.
Online Resources: I read various blogs and Stack Overflow discussions about Rust and ML libraries.

* Why I Could Not Run the Code *
The main issue was library incompatibility and type mismatches in the Burn framework.
Since Burn is a relatively new library, its documentation is not as extensive, making it difficult to debug errors.
Cargo dependencies were not properly resolved, which might require specifying exact versions or enabling additional features.

* What I Learned *
Rust's Type System is Strict
Unlike Python or other ML languages, Rust requires explicit type definitions, making it less flexible but more memory-safe.


* Resources Used *
Burn Documentation: https://docs.rs/burn
Rust Programming Language: https://doc.rust-lang.org/book/
Git Documentation: https://git-scm.com/doc
Stack Overflow: Various posts on Rust ML and Burn framework issues.

   
