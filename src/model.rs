pub struct Conv2D{
    input_size: u32,//input layers
    output_size: u32,//output layers
    filter: Vec<Vec<Vec<Vec<f32>>>>,
    bias: Vec<f32>
}

pub struct MaxPooling2D{
    pool_size: u32
}

pub struct Flatten{
    input_size: u32,
    output_size: u32
}

pub struct ReLU{
    input_size: u32,
    output_size: u32
}

enum LayerType{
    Conv2D,
    MaxPooling2D,
    Flatten,
    ReLU
}

impl Conv2D{
    pub fn new(input_size: u32, output_size: u32, filter: Vec<Vec<Vec<Vec<f32>>>>, bias: Vec<f32>) -> Conv2D {
        Conv2D{
            input_size: input_size,
            output_size: output_size,
            filter: filter,
            bias: bias
        }
    }
    pub fn forward(&self, input: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
        // the input is (layers, rows, cols));
        
        let mut output:  Vec<Vec<Vec<f32>>> 
            = vec![
                vec![
                    vec![0.0; input[0][0].len() - self.filter[0][0].len() + 1]
                    ;input[0].len() - self.filter[0][0][0].len() + 1
                ]
                ;self.output_size as usize
            ];

        println!("Output: {:?}", output);

        for i in 0..self.output_size{
            for j in 0..self.input_size{
                // i is the index of the output layer
                // j is the index of the filter group/input
                for filter in 0..self.filter[j as usize].len(){
                    for x in 0..(input[j as usize].len() - self.filter[j as usize][filter][0].len() + 1){
                        for y in 0..(input[j as usize][x as usize].len() - self.filter[j as usize][filter][0].len() + 1){
                            // x,y refers to the insert place of the filter to the output
                            for k in 0..self.filter[j as usize][filter].len(){
                                for l in 0..self.filter[j as usize][filter][k as usize].len(){
                                    output[i as usize][x as usize][y as usize] += 
                                        input[j as usize][x as usize + k as usize][y as usize + l as usize] * 
                                        self.filter[j as usize][filter][k as usize][l as usize];
                                }
                            }
                            output[i as usize][x as usize][y as usize] += self.bias[i as usize];
                        }
                    }
                }
            }
        }

        println!("Output: {:?}", output);

        output
    }
}

pub struct Model{
    layers: Vec<LayerType>    
}

#[cfg(test)]
mod tests {
    
    use super::*;
    
    #[test]
    fn Conv2D_test(){
        // these refer to the number of layers
        let input_size = 1;
        let output_size = 4;

        // 4 1 2 2
        let filter = vec![
            vec![//2x2
                vec![
                    vec![1.0, 0.5], 
                    vec![0.5, 1.0]
                ]
            ],
            vec![
                vec![
                    vec![1.0, 0.5], 
                    vec![0.5, 1.0]
                ]
            ],
            vec![
                vec![
                    vec![0.0, 0.5], 
                    vec![0.5, 0.0]
                ]
            ],
            vec![
                vec![
                    vec![1.0, 0.5], 
                    vec![0.5, 1.0]
                ]
            ]
        ];
        let bias = vec![0.0, 1.0, 2.0, 3.0];
        let conv2d = Conv2D::new(input_size, output_size, filter, bias);

        let input = vec![
            vec![
                vec![1.0, 1.0, 1.0, 1.0], 
                vec![1.0, 1.0, 1.0, 1.0], 
                vec![1.0, 1.0, 1.0, 1.0], 
                vec![1.0, 1.0, 1.0, 1.0]
            ]
        ];
        let output = conv2d.forward(&input);
        println!("H{:?}", output);
        assert_eq!(output, vec![
            vec![
                vec![3.0, 3.0, 3.0],
                vec![3.0, 3.0, 3.0],
                vec![3.0, 3.0, 3.0]
            ],
            vec![
                vec![4.0, 4.0, 4.0],
                vec![4.0, 4.0, 4.0],
                vec![4.0, 4.0, 4.0]
            ],
            vec![
                vec![2.0, 2.0, 2.0],
                vec![2.0, 2.0, 2.0],
                vec![2.0, 2.0, 2.0]
            ],
            vec![               
                vec![6.0, 6.0, 6.0],
                vec![6.0, 6.0, 6.0],
                vec![6.0, 6.0, 6.0]
            ]
        ], "Sample: {:?}", output);
    }
}