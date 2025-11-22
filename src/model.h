#pragma once

#include <torch/torch.h>

// Red neuronal para aproximar la función Q(s,a)
// Arquitectura: entrada -> fc1 (ReLU) -> fc2 (ReLU) -> fc3 -> salida
struct DQNImpl : torch::nn::Module {
  DQNImpl(int64_t dim_entrada, int64_t dim_oculta, int64_t dim_salida)
      : fc1(register_module("fc1", torch::nn::Linear(dim_entrada, dim_oculta))),
        fc2(register_module("fc2", torch::nn::Linear(dim_oculta, dim_oculta))),
        fc3(register_module("fc3", torch::nn::Linear(dim_oculta, dim_salida))) {}

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

TORCH_MODULE(DQN);
