#pragma once

#include <torch/torch.h>
#include <vector>
#include <random>

// Transición: experiencia (s, a, r, s', done) del agente
struct Transicion {
  std::vector<float> estado;
  int accion;
  float recompensa;
  std::vector<float> siguiente_estado;
  bool terminado;
};

// Buffer circular para almacenar experiencias (Experience Replay)
class ReplayBuffer {
public:
  ReplayBuffer(size_t capacidad);
  void agregar(const Transicion &t);
  size_t tamano() const;
  
  // Muestrea un lote aleatorio y retorna tensores para entrenamiento
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  muestrear(size_t tamano_lote);

private:
  std::vector<Transicion> buffer_;
  size_t capacidad_;
  size_t indice_escritura_;
  std::mt19937 generador_aleatorio_;
};
