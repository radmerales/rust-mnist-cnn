# Pure Rust CNN

This project implements a convolutional neural network from scratch using Rust. The project uses the UI library, Piston, to generate the drawing framework for the classifier. The training was done online through the use of Google Colab in Python.

`model.rs` implemented the following layers.

- `Conv2D`
- `MaxPooling2D`
- `ReLU`
- `FullyConnected`
- `argmax` and `softmax`

The `model.rs` file contained tests to make sure that the mentioned layers were working.

## Demonstration

Download the folder and run `cargo run`. The program will render a canvas where the user can draw on.

https://github.com/user-attachments/assets/004f4cd5-8f16-4cb2-996a-dd94affefebf

## Future Goals

- [ ] Implement a pure rust training
- [ ] Improve speed. Implement SIMD or GPU Programming
