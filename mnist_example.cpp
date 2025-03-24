#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <tuple>
#define HIDDEN_SIZE 64
#define BATCH_SIZE 256
// Define a custom dataset class for MNIST
class MNISTDataset : public torch::data::datasets::Dataset<MNISTDataset> {
  private:
      torch::Tensor data_, targets_;
  public:
      // Constructor: load data from file saved by torch.save in Python
      explicit MNISTDataset(torch::Tensor data, torch::Tensor targets): data_(data), targets_(targets) {
          //torch::load(data_, data_file_path);
          //torch::load(targets_, targets_file_path);
      }
      // Override get() to return an example (data sample and its label)
      torch::data::Example<> get(size_t index) override {
          // For many models, you might want to add a channel dimension.
          // Here we unsqueeze to add the channel dimension (1, 28, 28)
          torch::Tensor sample = (data_[index].unsqueeze(0).to(torch::kFloat32))/255;
          torch::Tensor label = targets_[index];
          return {sample, label};
      }
      // Override size() to return the number of samples in the dataset
      torch::optional<size_t> size() const override {
          return data_.size(0);
      }
  };

// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, HIDDEN_SIZE));
    fc2 = register_module("fc2", torch::nn::Linear(HIDDEN_SIZE, HIDDEN_SIZE));
    fc3 = register_module("fc3", torch::nn::Linear(HIDDEN_SIZE, HIDDEN_SIZE/2));
    fc4 = register_module("fc4", torch::nn::Linear(HIDDEN_SIZE/2, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    //x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::relu(fc3->forward(x));
    x = fc4->forward(x);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};

int main() {
  // Create a new Net.
  auto net = std::make_shared<Net>();
  // Path to the file saved by torch.save in Python
  torch::jit::script::Module data_module, test_data_module;
  data_module = torch::jit::load("mnist_training_module.pt");
  test_data_module = torch::jit::load("mnist_testing_module.pt");
  
  //get training data
  std::vector<torch::jit::IValue> inputs;
  auto output = data_module.forward(inputs).toTuple();
  torch::Tensor data = output->elements()[0].toTensor();
  torch::Tensor targets = output->elements()[1].toTensor();
  std::cout << "Loaded MNIST data tensor with shape: " << data.sizes() << std::endl;
  std::cout << "Loaded MNIST targets tensor with shape: " << targets.sizes() << std::endl;
  // Create an instance of the custom MNIST dataset
  auto dataset = MNISTDataset(data, targets)
                      .map(torch::data::transforms::Stack<>());
  // Create a DataLoader with chosen batch size
  auto data_loader = torch::data::make_data_loader(std::move(dataset), /*batch_size=*/BATCH_SIZE);

  //get testing data
  output = test_data_module.forward(inputs).toTuple();
  torch::Tensor test_data = output->elements()[0].toTensor();
  torch::Tensor test_targets = output->elements()[1].toTensor();
  std::cout << "Loaded MNIST test data tensor with shape: " << test_data.sizes() << std::endl;
  std::cout << "Loaded MNIST test targets tensor with shape: " << test_targets.sizes() << std::endl;
  // Create an instance of the custom MNIST dataset
  auto test_dataset = MNISTDataset(test_data, test_targets)
                      .map(torch::data::transforms::Stack<>());
  // Create a DataLoader with chosen batch size
  auto test_data_loader = torch::data::make_data_loader(std::move(test_dataset), /*batch_size=*/BATCH_SIZE);

  // // Create a multi-threaded data loader for the MNIST dataset.
  // auto data_loader = torch::data::make_data_loader(
  //     torch::data::datasets::MNIST("mnist_training_data.pt").map(
  //         torch::data::transforms::Stack<>()),
  //     /*batch_size=*/64);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::Adam optimizer(net->parameters(), /*lr=*/0.001);
  torch::nn::CrossEntropyLoss loss_function;
  //train
  for (size_t epoch = 1; epoch <= 25; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    for (auto& batch : *data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = net->forward(batch.data);
      //std::cout << batch.data[0] << std::endl;
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = loss_function(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        torch::save(net, "net.pt");
      }
    }
  }
  
  //test
  //create softmax function to apply to predicted labels


  // Iterate the data loader to yield batches from the dataset.
  double batch_error = 0;
  double global_batch_error = 0;
  for (auto& batch : *test_data_loader) {
    // Execute the model on the input data.
    torch::Tensor prediction = net->forward(batch.data);
    prediction = torch::softmax(prediction, /*dim=*/1);
    torch::Tensor predicted_index = torch::argmax(prediction, /*dim=*/1);
    torch::Tensor correct = batch.target == predicted_index;
    correct = correct.to(torch::kInt);
    //std::cout << "Accuracy tensor with shape: " << correct.sizes() << std::endl;
    batch_error = (correct.sum().item<double>())*100;
    global_batch_error += batch_error;
    //std::cout << batch.data[0] << std::endl;
  }

std::cout << "test accuracy: " << global_batch_error/10000 << "%" << std::endl;
}
