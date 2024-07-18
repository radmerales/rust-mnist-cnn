extern crate piston_window;
extern crate serde_json;

pub mod model;

use std::fs;
use piston_window::*;

fn draw_canvas(state: &Vec<Vec<f32>>, c: &Context, g: &mut G2d){
    for i in 0..28{
        for j in 0..28{
            rectangle(
                [0.0, 0.0, 0.0, state[i as usize][j as usize]],
                [
                    20.0 * (i as f64),
                    20.0 * (j as f64),
                    20.0, 20.0
                ],
                c.transform, g
            );
        }
    }
}

fn generate_conv2d(json: &serde_json::Value, name: &str) -> model::Conv2D{
    let mut weight: String = name.to_owned();
    weight.push_str(".weight");
    let mut bias: String = name.to_owned();
    bias.push_str(".bias");
    
    let weight = json[weight].clone();
    let mut weight_values = vec![
                                vec![
                                    vec![
                                        vec![0.0; weight[0][0][0].as_array().unwrap().len()]
                                    ; weight[0][0].as_array().unwrap().len()]
                                ; weight[0].as_array().unwrap().len()]
                            ; weight.as_array().unwrap().len()
                            ];
        
    for i in 0..weight.as_array().unwrap().len(){
        for j in 0..weight[i].as_array().unwrap().len(){
            for k in 0..weight[i][j].as_array().unwrap().len(){
                for l in 0..weight[i][j][k].as_array().unwrap().len(){
                    weight_values[i][j][k][l] = weight[i][j][k][l].as_f64().unwrap() as f32;
                }
            }
        }
    }

    let bias = json[bias].clone();
    let mut bias_values = vec![0.0; bias.as_array().unwrap().len()];
    for i in 0..bias.as_array().unwrap().len(){
        bias_values[i] = bias[i].as_f64().unwrap() as f32;
    }

    model::Conv2D::new(
        weight[0].as_array().unwrap().len() as u32,
        weight.as_array().unwrap().len() as u32,
        weight_values, 
        bias_values
    )
}

fn generate_fully_connected(json: &serde_json::Value, name: &str) -> model::FullyConnected{
    let mut weight: String = name.to_owned();
    weight.push_str(".weight");
    let mut bias: String = name.to_owned();
    bias.push_str(".bias");
    
    let weight = json[weight].clone();
    let mut weight_values = vec![
                                vec![0.0; weight[0].as_array().unwrap().len()]
                                ; weight.as_array().unwrap().len()
                            ];
    
    for i in 0..weight.as_array().unwrap().len(){
        for j in 0..weight[i].as_array().unwrap().len(){
            weight_values[i][j] = weight[i][j].as_f64().unwrap() as f32;
        }
    }

    let bias = json[bias].clone();
    let mut bias_values = vec![0.0; bias.as_array().unwrap().len()];
    for i in 0..bias.as_array().unwrap().len(){
        bias_values[i] = bias[i].as_f64().unwrap() as f32;
    }

    model::FullyConnected::new(
        weight[0].as_array().unwrap().len() as u32,
        weight.as_array().unwrap().len() as u32,
        weight_values, 
        bias_values
    )
}

#[derive(Debug)]
struct CNN{
    input_size: u32,
    output_size: u32,
    conv1: model::Conv2D,
    conv2: model::Conv2D,
    fc: model::FullyConnected
}

impl CNN{
    fn new(input_size: u32, output_size: u32, conv1: model::Conv2D, conv2: model::Conv2D, fc: model::FullyConnected) -> CNN{
        CNN{
            input_size: input_size,
            output_size: output_size,
            conv1: conv1,
            conv2: conv2,
            fc: fc
        }
    }

    fn forward(&self, img: &Vec<Vec<Vec<f32>>>) -> u32{
        let img = self.conv1.forward(img);
        let img = model::ReLU::forward(&img);
        let pool2 = model::MaxPooling2D::new(2);
        let img = pool2.forward(&img);
        
        let img = self.conv2.forward(&img);
        let img = model::ReLU::forward(&img);
        let img = pool2.forward(&img);

        let img = model::Flatten::forward(&img);
        let img = self.fc.forward(&img);
        
        let img = model::softmax(&img);
        model::argmax(&img) as u32
    }
}

fn print_screen(erase: &bool){
    println!("{}[2J", 27 as char);

    println!("Draw Rust!");
    println!("Current Mode: {}", {
        if *erase{
            "Erase"
        }
        else{
            "Write"
        }
    });
    println!("Press D to draw, E to erase, C to clear, P to predict");

}

fn main() {
    let mut window: PistonWindow = 
        WindowSettings::new("Draw Rust!", [540, 540])
        .exit_on_esc(true).build().unwrap();
    let mut draw = false;
    let mut erase = false;
    let fill = 1;

    let mut state: Vec<Vec<f32>> = vec![vec![0.0; 28]; 28];
    
    //let model = predict::CNN::new();

    let file_result = fs::File::open("./src/assets/model.json");
    let file = match file_result {
        Ok(file) => file,
        Err(e) => {
            println!("Error: {}", e);
            return;
        }
    };


    let json: serde_json::Value = serde_json::from_reader(file)
        .expect("file should be proper JSON");

    let conv1 = generate_conv2d(&json, "conv1");
    let conv2 = generate_conv2d(&json, "conv2");
    let fc = generate_fully_connected(&json, "fc1");

    let cnn = CNN::new(1, 10, conv1, conv2, fc);
    

    while let Some(e) = window.next() {
        
        window.draw_2d(&e, |c, g, _device| {
            clear([1.0; 4], g);
            
            draw_canvas(&state, &c, g);
            //buttons.push(draw_button(&vec![0.0, 600.0, 100.0, 30.0], &state, &c, g, &mut glyphs));
        });

        if let Some(button) = e.press_args() {
            if button == Button::Keyboard(Key::E) {
                erase = true;
                print_screen(&erase);
                //println!("Erase Mode");
            }
            else if button == Button::Keyboard(Key::D){
                erase = false;
                print_screen(&erase);
                //println!("Write Mode");
            }
            else if button == Button::Keyboard(Key::C){
                // clear everything, back to draw
                erase = false;
                state = vec![vec![0.0; 28]; 28];
                print_screen(&erase);
                //println!("Clear - Write Mode");
            }
            else if button == Button::Keyboard(Key::P){
                //println!("{:?}", state);
                let mut convert = vec![vec![0.0; 28]; 28];
                for i in 0..28{
                    for j in 0..28{
                        convert[i][j] = (state[j][i]-0.5)/0.5;
                    }
                }

                let wrapper = vec![convert.clone()];
                let result = cnn.forward(&wrapper);
                print_screen(&erase);
                println!("Predicted: {}", result);
            }
        };

        if let Some(button) = e.press_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = true;
                //println!("Mouse Press");

            }
        };
        if let Some(button) = e.release_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = false;
                //println!("Mouse up");
            }
        };

        if draw {
            if let Some(pos) = e.mouse_cursor_args() {
                let (x, y) = (((pos[0] as f32)/20.0).floor(), ((pos[1] as f32)/20.0).floor());
                if x < 28.0 && x >= 0.0 && y < 28.0 && y >= 0.0{
                    if erase{
                        state[x as usize][y as usize] = 0.0;
                        if fill == 2{
                            let add_x = vec![1.0, -1.0, 0.0, 0.0];
                            let add_y = vec![0.0, 0.0, -1.0, 1.0];
                            for i in 0..4{
                                let new_x = x + add_x[i];
                                let new_y = y + add_y[i];
                                if new_x < 28.0 && new_x >= 0.0 && new_y < 28.0 && new_y >= 0.0{
                                    state[new_x as usize][new_y as usize] = 0.0;
                                }
                            }
                        }
                    }
                    else{
                        state[x as usize][y as usize] = 1.0;
                        if fill == 2{
                            let add_x = vec![1.0, -1.0, 0.0, 0.0];
                            let add_y = vec![0.0, 0.0, -1.0, 1.0];
                            for i in 0..4{
                                let new_x = x + add_x[i];
                                let new_y = y + add_y[i];
                                if new_x < 28.0 && new_x >= 0.0 && new_y < 28.0 && new_y >= 0.0{
                                    state[new_x as usize][new_y as usize] = std::cmp::max((state[new_x as usize][new_y as usize]*100.0) as u32, 99_u32) as f32 / 100.0;
                                }
                            }
                        }
                    }//println!("{}, {}", (x/20.0).floor(),(y/20.0).floor());
                }
            };
        }
    }
}