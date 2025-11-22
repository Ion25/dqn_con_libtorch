#pragma once

#include <torch/torch.h>
#include "model.h"
#include "replay_buffer.h"

// Agente DQN: gestiona redes policy/target, optimización y experiencia
class DQNAgent {
public:
  DQNAgent(int dim_estado, int dim_accion, int64_t neuronas_ocultas=128, 
           float tasa_aprendizaje=1e-3, bool usar_double_dqn=true);
  
  int seleccionar_accion(const torch::Tensor &estado, float epsilon);
  void recordar(const Transicion &t);
  void optimizar();
  void actualizar_red_objetivo();
  void a_dispositivo(torch::Device dispositivo);
  void guardar(const std::string &ruta);
  void cargar(const std::string &ruta);

private:
  DQN red_policy_{nullptr};        // Red que se entrena
  DQN red_objetivo_{nullptr};      // Red objetivo (parámetros fijos temporalmente)
  std::unique_ptr<ReplayBuffer> memoria_;
  torch::optim::Adam optimizador_;
  int dim_accion_;
  torch::Device dispositivo_;
  size_t tamano_lote_;
  float gamma_;                     // Factor de descuento
  bool usar_double_dqn_;
};
