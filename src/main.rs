extern crate piston_window;
extern crate gfx_device_gl;
extern crate gfx_text;
extern crate find_folder;

pub mod model;

use piston_window::*;

struct ClickableButton{
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    color: [f32; 4],
    text: String
}

fn draw_canvas(state: &Vec<Vec<bool>>, c: &Context, g: &mut G2d){
    let black = [0.0, 0.0, 0.0, 1.0];
    let white = [0.0, 0.0, 0.0, 0.0];
    for i in 0..28{
        for j in 0..28{
            rectangle(
                {
                    if state[i][j]{
                        white
                    }
                    else{
                        black
                    }
                },
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

fn draw_button(pos: &Vec<f64> ,state: &Vec<Vec<bool>>, c: &Context, g: &mut G2d, glyphs: &mut Glyphs) -> ClickableButton{
    let black = [0.0, 0.0, 0.0, 1.0];
    let white = [0.0, 0.0, 0.0, 0.0];
    let button = ClickableButton{
        x: pos[0],
        y: pos[1],
        width: pos[2],
        height: pos[3],
        color: [1.0, 0.0, 0.0, 1.0],
        text: "Erase".to_string()
    };
    rectangle(
        button.color,
        [button.x, button.y, button.width, button.height],
        c.transform, g
    );
    let size = 20.0;
    
    // for horizontal center
    //  - consider length, consider font size, consider width of rectangle
    // for vertical center
    //  - consider font size, consider height of rectangle
    
    text::Text::new_color([1.0, 1.0, 1.0, 1.0], size as u32).draw(
        &button.text,
        glyphs,
        &c.draw_state,
        c.transform.trans(button.x + 10.0, size + button.y + (button.height - size)/2.0),
        g,
    )
    .unwrap();

    button
}

fn main() {
    let mut window: PistonWindow = 
        WindowSettings::new("Draw Rust!", [540, 540])
        .exit_on_esc(true).build().unwrap();
    let mut draw = false;
    let mut erase = false;

    // Load the font
    let assets = find_folder::Search::ParentsThenKids(3, 3).for_folder("assets").unwrap();
    let ref font_path = assets.join("Bebas-Neue.ttf");
    let factory = window.factory.clone();
    let mut glyphs = window.load_font(font_path).unwrap();
    
    let mut state: Vec<Vec<bool>> = vec![vec![false; 28]; 28];
    
    //let model = predict::CNN::new();

    while let Some(e) = window.next() {
        print!("{}[2J", 27 as char);
        println!("Draw Rust!");
        println!("Current Mode: {}", {
            if erase{
                "Erase"
            }
            else{
                "Write"
            }
        });
        println!("Press D to draw, E to erase, C to clear, P to predict");
        
        let mut buttons: Vec<ClickableButton> = vec![];
        window.draw_2d(&e, |c, g, device| {
            clear([1.0; 4], g);
            
            // ! Fix the pass to draw canvas, it should be a reference
            draw_canvas(&state, &c, g);
            //buttons.push(draw_button(&vec![0.0, 600.0, 100.0, 30.0], &state, &c, g, &mut glyphs));
            
            // Update glyphs
            glyphs.factory.encoder.flush(device);
        });

        if let Some(button) = e.press_args() {
            if button == Button::Keyboard(Key::E) {
                erase = true;
                println!("Erase Mode");
            }
            else if button == Button::Keyboard(Key::D){
                erase = false;
                println!("Write Mode");
            }
            else if button == Button::Keyboard(Key::C){
                // clear everything, back to draw
                erase = false;
                state = vec![vec![false; 28]; 28];
                println!("Clear - Write Mode");
            }
        };

        if let Some(button) = e.press_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = true;
                println!("Mouse Press");

                for buttones in buttons.iter(){
                    println!("Button: {}, {}, {}, {}", buttones.x, buttones.y, buttones.width, buttones.height);
                    if buttones.x < e.mouse_cursor_args().unwrap()[0] && e.mouse_cursor_args().unwrap()[0] < buttones.x + buttones.width && buttones.y < e.mouse_cursor_args().unwrap()[1] && e.mouse_cursor_args().unwrap()[1] < buttones.y + buttones.height{
                        println!("Erase Button Pressed: {}", erase);
                        break;
                    }
                }

            }
        };
        if let Some(button) = e.release_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = false;
                println!("Mouse up");
            }
        };

        if draw {
            if let Some(pos) = e.mouse_cursor_args() {
                let (x, y) = (pos[0] as f32, pos[1] as f32);
                if (x/20.0).floor() < 28.0 && x > 0.0 && (y/20.0).floor() < 28.0 && y > 0.0{
                    state[(x/20.0).floor() as usize][(y/20.0).floor() as usize] = !erase;
                    //println!("{}, {}", (x/20.0).floor(),(y/20.0).floor());
                }
            };
        }
    }
}