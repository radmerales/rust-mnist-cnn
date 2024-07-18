#[derive(Debug)]
pub struct Conv2D{
    input_size: u32,//input layers
    output_size: u32,//output layers
    filter: Vec<Vec<Vec<Vec<f32>>>>,
    bias: Vec<f32>
}

#[derive(Debug)]
pub struct MaxPooling2D{
    pool_size: u32
}

#[derive(Debug)]
pub struct Flatten{
    input_size: u32,
    output_size: u32
}

#[derive(Debug)]
pub struct ReLU{
    input_size: u32,
    output_size: u32
}

#[derive(Debug)]
pub struct FullyConnected{
    input_size: u32,
    output_size: u32,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>
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

        for i in 0..self.output_size{
            //println!("Filters: {:?}", self.filter[i as usize]);
            for j in 0..self.input_size{
                //println!("Filter for input {}: {:?}", j, self.filter[i as usize][j as usize]);
                // i is the index of the output layer
                // j is the index of the filter group/input
                for x in 0..(input[j as usize].len() - self.filter[i as usize][j as usize][0].len() + 1){
                    for y in 0..(input[j as usize][x as usize].len() - self.filter[i as usize][j as usize][0].len() + 1){
                        // x,y refers to the insert place of the filter to the output
                        for k in 0..self.filter[i as usize][j as usize].len(){
                            for l in 0..self.filter[i as usize][j as usize][k as usize].len(){
                                output[i as usize][x as usize][y as usize] += 
                                    input[j as usize][x as usize + k as usize][y as usize + l as usize] * 
                                    self.filter[i as usize][j as usize][k as usize][l as usize];
                            }
                        }
                        output[i as usize][x as usize][y as usize] += self.bias[i as usize];
                    }
                }
            }
        }
        output
    }
}

impl MaxPooling2D{
    pub fn new(pool_size: u32) -> MaxPooling2D{
        MaxPooling2D{
            pool_size: pool_size
        }
    }

    pub fn forward(&self, input: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>>{
        // # MaxPooling2D
        // Input: (layers, rows, cols)
        // Output: (layers, floor(rows/pool_size), floor(cols/pool_size))
        let mut output: Vec<Vec<Vec<f32>>> = vec![
            vec![
                vec![0.0; input[0][0].len() / self.pool_size as usize]
                ;input[0].len() / self.pool_size as usize
            ]
            ;input.len() as usize
        ];

        for layer in 0..input.len(){
            for x in 0..input[layer].len() / self.pool_size as usize{
                for y in 0..input[layer][x].len() / self.pool_size as usize{
                    let mut max = 0.0;
                    for ii in 0..self.pool_size{
                        let i = ii as usize;
                        for jj in 0..self.pool_size{
                            let j = jj as usize;
                            if input[layer][x * self.pool_size as usize + i][y * self.pool_size as usize + j] > max{
                                max = input[layer][x * self.pool_size as usize + i][y * self.pool_size as usize + j];
                            }
                        }
                    }
                    output[layer][x][y] = max;
                }
            }
        }
        output
    }
}

impl Flatten{
    pub fn new(input_size: u32, output_size: u32) -> Flatten{
        Flatten{
            input_size: input_size,
            output_size: output_size
        }
    }

    pub fn forward(img: &Vec<Vec<Vec<f32>>>) -> Vec<f32>{
        let mut output: Vec<f32> = vec![0.0; img.len() * img[0].len() * img[0][0].len()];
        let mut index = 0;
        for i in 0..img.len(){
            for j in 0..img[i].len(){
                for k in 0..img[i][j].len(){
                    output[index] = img[i][j][k];
                    index += 1;
                }
            }
        }
        output
    }
}

impl ReLU{
    pub fn new(input_size: u32, output_size: u32) -> ReLU{
        ReLU{
            input_size: input_size,
            output_size: output_size
        }
    }

    pub fn forward(input: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>>{
        let mut output: Vec<Vec<Vec<f32>>> = vec![
            vec![
                vec![0.0; input[0][0].len()]
                ;input[0].len()
            ]
            ;input.len()
        ];
        for i in 0..input.len(){
            for j in 0..input[i].len(){
                for k in 0..input[i][j].len(){
                    output[i][j][k] = if input[i][j][k] > 0.0 {input[i][j][k]} else {0.0};
                }
            }
        }
        output
    }
}

impl FullyConnected{
    pub fn new(input_size: u32, output_size: u32, weights: Vec<Vec<f32>>, bias: Vec<f32>) -> FullyConnected{
        FullyConnected{
            input_size: input_size,
            output_size: output_size,
            weights: weights,
            bias: bias
        }
    }
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32>{
        let mut output: Vec<f32> = vec![0.0; self.output_size as usize];
        for i in 0..self.output_size{
            for j in 0..self.input_size{
                output[i as usize] += input[j as usize] * self.weights[i as usize][j as usize];
            }
            output[i as usize] += self.bias[i as usize];
        }
        output
    }
}

pub fn argmax(input: &Vec<f32>) -> usize{
    let mut max = 0.0;
    let mut index = 0;
    for i in 0..input.len(){
        if input[i] > max{
            max = input[i];
            index = i;
        }
    }
    index
}

pub fn softmax(input: &Vec<f32>) -> Vec<f32>{
    let mut output: Vec<f32> = vec![0.0; input.len()];
    let mut sum = 0.0;
    for i in 0..input.len(){
        sum += input[i].exp();
    }
    for i in 0..input.len(){
        output[i] = input[i].exp() / sum;
    }
    output
}

#[cfg(test)]
mod tests {
    
    use super::*;
    
    #[test]
    fn conv2d_test(){
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
                    vec![0.0, 0.5], 
                    vec![0.0, 1.0]
                ]
            ]
        ];
        let bias = vec![0.0, 1.0, 1.0, 0.0];
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
                vec![1.5, 1.5, 1.5],
                vec![1.5, 1.5, 1.5],
                vec![1.5, 1.5, 1.5]
            ]
        ], "Sample: {:?}", output);
    }

    #[test]
    fn max_pooling_2d_test(){
        let pool_size = 2;
        let maxpooling2d = MaxPooling2D::new(pool_size);

        let input = vec![
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
                vec![13.0, 14.0, 15.0, 16.0]
            ]
        ];
        let output = maxpooling2d.forward(&input);
        assert_eq!(output, vec![
            vec![
                vec![6.0, 8.0],
                vec![14.0, 16.0]
            ]
        ], "Sample: {:?}", output);

        let input = vec![
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![6.0, 7.0, 8.0, 9.0, 10.0],
                vec![11.0, 12.0, 13.0, 14.0, 15.0],
                vec![16.0, 17.0, 18.0, 19.0, 20.0],
                vec![21.0, 22.0, 23.0, 24.0, 25.0]
            ]
        ];
        let output = maxpooling2d.forward(&input);
        assert_eq!(output, vec![
            vec![
                vec![7.0, 9.0],
                vec![17.0, 19.0]
            ]
        ], "Sample: {:?}", output);
    }

    #[test]
    fn flatten_test(){
        let input = vec![
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0]
            ]
        ];
        let output = Flatten::forward(&input);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], "Sample: {:?}", output);
    }

    #[test]
    fn relu_test(){
        let input = vec![
            vec![
                vec![1.0, -2.0, 3.0],
                vec![-4.0, 5.0, -6.0],
                vec![7.0, -8.0, 9.0]
            ],
            vec![
                vec![-1.0, 2.0, -3.0],
                vec![4.0, -5.0, 6.0],
                vec![-7.0, 8.0, -9.0]
            ]
        ];
        let output = ReLU::forward(&input);
        assert_eq!(output, vec![
            vec![
                vec![1.0, 0.0, 3.0],
                vec![0.0, 5.0, 0.0],
                vec![7.0, 0.0, 9.0]
            ],
            vec![
                vec![0.0, 2.0, 0.0],
                vec![4.0, 0.0, 6.0],
                vec![0.0, 8.0, 0.0]
            ]
        ], "Sample: {:?}", output);
    }

    #[test]
    fn fully_connected_test(){
        let input_size = 5;
        let output_size = 2;
        let weights = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0]
        ];
        let bias = vec![1.0, 2.0];
        let fully_connected = FullyConnected::new(input_size, output_size, weights, bias);

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = fully_connected.forward(&input);
        assert_eq!(output, vec![56.0, 132.0], "Sample: {:?}", output);
    }

    #[test]
    fn model_test(){
        let x = vec![
            vec![
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        ];

        let conv2d1 = Conv2D::new(1, 2, 
            vec![
                vec![
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
                ]
            ], vec![0.0, 1.0]);
        let x = conv2d1.forward(&x);
        println!("Conv2D: {:?}", x);
        
        let maxpooling2d1 = MaxPooling2D::new(2);
        let x = maxpooling2d1.forward(&x);
        println!("MaxPooling2D: {:?}", x);
        
        assert!(1==2);
    }
}