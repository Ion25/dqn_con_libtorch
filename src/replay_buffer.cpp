#include "replay_buffer.h"
#include <algorithm>

ReplayBuffer::ReplayBuffer(size_t capacidad)
    : capacidad_(capacidad), indice_escritura_(0), 
      generador_aleatorio_(std::random_device{}()) {
  buffer_.reserve(capacidad_);
}

void ReplayBuffer::agregar(const Transicion &t) {
  if (buffer_.size() < capacidad_) {
    buffer_.push_back(t);
  } else {
    // Sobrescribir la más antigua (buffer circular)
    buffer_[indice_escritura_] = t;
    indice_escritura_ = (indice_escritura_ + 1) % capacidad_;
  }
}

size_t ReplayBuffer::tamano() const { return buffer_.size(); }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ReplayBuffer::muestrear(size_t tamano_lote) {
  std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);
  std::vector<torch::Tensor> estados, siguientes_estados;
  std::vector<int64_t> acciones;
  std::vector<float> recompensas;
  std::vector<uint8_t> terminados;

  for (size_t i = 0; i < tamano_lote; ++i) {
    const auto &t = buffer_[dist(generador_aleatorio_)];
    estados.push_back(torch::from_blob((void *)t.estado.data(), 
                      {(int64_t)t.estado.size()}, torch::kFloat).clone());
    siguientes_estados.push_back(torch::from_blob((void *)t.siguiente_estado.data(), 
                                  {(int64_t)t.siguiente_estado.size()}, torch::kFloat).clone());
    acciones.push_back(t.accion);
    recompensas.push_back(t.recompensa);
    terminados.push_back(t.terminado ? 1 : 0);
  }

  auto tensor_estados = torch::stack(estados);
  auto tensor_siguientes_estados = torch::stack(siguientes_estados);
  auto tensor_acciones = torch::from_blob(acciones.data(), 
                         {(int64_t)acciones.size()}, torch::kLong).clone();
  auto tensor_recompensas = torch::from_blob(recompensas.data(), 
                            {(int64_t)recompensas.size()}, torch::kFloat).clone();
  auto tensor_terminados = torch::from_blob(terminados.data(), 
                           {(int64_t)terminados.size()}, torch::kUInt8).to(torch::kFloat).clone();

  return {tensor_estados, tensor_acciones, tensor_recompensas, 
          tensor_siguientes_estados, tensor_terminados};
}
