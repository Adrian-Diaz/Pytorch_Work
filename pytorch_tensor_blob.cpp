#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

/* MNIST model loading example; loaded model has input dim of 784*/

struct model_wrapper{
  int input_size = 784;
  torch::TensorOptions tensor_options;
  torch::jit::script::Module model;
  std::string file_path;
};

extern "C" {
  model_wrapper* allocate_model_struct();
  void delete_model_struct(model_wrapper *container);
  void setup_model(model_wrapper *container);
  void forward_propagate_model(float* data, int batch_size, model_wrapper *container);
}


int main() {
  int batch_size = 4;
  int size = 784*batch_size;
  float* data = new float[size];
  //populate with arbitrary data for now
  for(int init = 0; init < size; init++){
     data[init] = 3.14*static_cast<float>(init);
  }

  //wrapper struct (can probably call in Fortran interface at least through a pointer)
  model_wrapper* model_ptr = allocate_model_struct();

  //load parameters
  setup_model(model_ptr);
  forward_propagate_model(data, batch_size, model_ptr);
  delete[] data;
  delete_model_struct(model_ptr);
}

//allocate model wrapper struct and retunr pointer for Fortran interface
model_wrapper* allocate_model_struct(){

  model_wrapper* ptr = new model_wrapper;
  return ptr;
}

void delete_model_struct(model_wrapper *container){

  delete container;

}

//loads model parameters and sets tensor options
void setup_model(model_wrapper *container){

  //model parameters file path
  container->file_path = "model_saved.pt";
  //default tensors are row major and float32 type; use kFloat64 for double
  container->tensor_options = torch::TensorOptions().dtype(torch::kFloat32);
  //load model parameters in from torchscript save (sets model object for us)
  container->model = torch::jit::load(container->file_path);
  container->model.eval();
}

//forward propagate using model
void forward_propagate_model(float* data, int batch_size, model_wrapper *container){
  
  auto options = container->tensor_options;
  {//code block for inference mode
    torch::InferenceMode guard(true);

    //array dims should reflect the loaded model
    torch::Tensor tensor = torch::from_blob(data, {batch_size,784}, options);

    //allocate column major (fortran contiguous) tensor; next array argument is strides
    //torch::Tensor col_tensor = torch::from_blob(data, {batch_size, 784}, {1,batch_size}, options);
    torch::Tensor col_tensor = torch::from_blob(data, {784, batch_size}, options);
    auto new_col_tensor = col_tensor.permute({1,0}).contiguous();
  
    // std::cout << "row major input tensor" << std::endl;
    // std::cout << tensor << std::endl;
    // std::cout << "column major input tensor" << std::endl;
    std::cout << new_col_tensor << std::endl;
    std::cout << new_col_tensor.strides() << std::endl;

    std::vector<torch::jit::IValue> tensor_inputs, col_tensor_inputs;
    //tensor_inputs.push_back(col_tensor);
    tensor_inputs.push_back(tensor);
    col_tensor_inputs.push_back(new_col_tensor);

    //forward propagate model
    torch::jit::IValue output_data = container->model.forward(tensor_inputs);
    torch::jit::IValue col_output_data = container->model.forward(col_tensor_inputs);
    torch::Tensor output_tensor = output_data.toTensor();
    torch::Tensor col_output_tensor = col_output_data.toTensor();

    //std::cout << "row major tensor" << std::endl;
    //std::cout << tensor << std::endl;
    //std::cout << "column major tensor" << std::endl;
    //std::cout << col_tensor << std::endl;
    std::cout << "model output tensor" << std::endl;
    std::cout << output_tensor << std::endl;
    std::cout << "model output col tensor" << std::endl;
    std::cout << col_output_tensor << std::endl;
  }
}
