#include "dqn_agent.h"
#include <iostream>

DQNAgent::DQNAgent(int dim_estado, int dim_accion, int64_t neuronas_ocultas, 
                   float tasa_aprendizaje, bool usar_double_dqn)
    : red_policy_(DQN(dim_estado, neuronas_ocultas, dim_accion)),
      red_objetivo_(DQN(dim_estado, neuronas_ocultas, dim_accion)),
      memoria_(std::make_unique<ReplayBuffer>(100000)),
      optimizador_(red_policy_->parameters(), torch::optim::AdamOptions(tasa_aprendizaje)),
      dim_accion_(dim_accion), dispositivo_(torch::kCPU), tamano_lote_(64), gamma_(0.99f),
      usar_double_dqn_(usar_double_dqn) {
  // Copiar parámetros de red policy a red objetivo
  torch::NoGradGuard no_grad;
  auto params_policy = red_policy_->named_parameters();
  auto params_objetivo = red_objetivo_->named_parameters();
  for (auto& param : params_policy) {
    params_objetivo[param.key()].copy_(param.value());
  }
}

int DQNAgent::seleccionar_accion(const torch::Tensor &estado, float epsilon) {
  // Epsilon-greedy: exploración vs explotación
  if (((double)rand() / RAND_MAX) < epsilon) {
    return rand() % dim_accion_;
  } else {
    red_policy_->eval();
    torch::NoGradGuard no_grad;
    auto s = estado.to(dispositivo_).unsqueeze(0);
    auto valores_q = red_policy_->forward(s);
    auto accion = valores_q.argmax(1).item<int>();
    return accion;
  }
}

void DQNAgent::recordar(const Transicion &t) { memoria_->agregar(t); }

void DQNAgent::optimizar() {
  if (memoria_->tamano() < tamano_lote_) return;
  
  red_policy_->train();
  auto lote = memoria_->muestrear(tamano_lote_);
  auto estados = std::get<0>(lote).to(dispositivo_);
  auto acciones = std::get<1>(lote).to(dispositivo_);
  auto recompensas = std::get<2>(lote).to(dispositivo_);
  auto siguientes_estados = std::get<3>(lote).to(dispositivo_);
  auto terminados = std::get<4>(lote).to(dispositivo_);

  // Valores Q actuales para las acciones tomadas
  auto valores_q = red_policy_->forward(estados).gather(1, acciones.unsqueeze(1)).squeeze(1);

  // Calcular valores Q esperados (target)
  torch::Tensor q_esperado;
  {
    torch::NoGradGuard no_grad;
    torch::Tensor siguientes_valores_q;
    
    if (usar_double_dqn_) {
      // Double DQN: seleccionar con policy, evaluar con objetivo
      auto siguientes_acciones = red_policy_->forward(siguientes_estados).argmax(1);
      siguientes_valores_q = red_objetivo_->forward(siguientes_estados)
                              .gather(1, siguientes_acciones.unsqueeze(1)).squeeze(1);
    } else {
      // DQN estándar
      siguientes_valores_q = std::get<0>(red_objetivo_->forward(siguientes_estados).max(1));
    }
    
    q_esperado = recompensas + gamma_ * siguientes_valores_q * (1 - terminados);
  }

  // Huber loss (smooth L1) para robustez
  auto perdida = torch::smooth_l1_loss(valores_q, q_esperado);

  optimizador_.zero_grad();
  perdida.backward();
  // Recorte de gradientes para estabilidad
  torch::nn::utils::clip_grad_norm_(red_policy_->parameters(), 10.0);
  optimizador_.step();
}

void DQNAgent::actualizar_red_objetivo() {
  torch::NoGradGuard no_grad;
  auto params_policy = red_policy_->named_parameters();
  auto params_objetivo = red_objetivo_->named_parameters();
  for (auto& param : params_policy) {
    params_objetivo[param.key()].copy_(param.value());
  }
}

void DQNAgent::a_dispositivo(torch::Device dispositivo) {
  dispositivo_ = dispositivo;
  red_policy_->to(dispositivo_);
  red_objetivo_->to(dispositivo_);
}

void DQNAgent::guardar(const std::string &ruta) { 
  torch::save(red_policy_, ruta); 
}

void DQNAgent::cargar(const std::string &ruta) { 
  torch::load(red_policy_, ruta); 
}
