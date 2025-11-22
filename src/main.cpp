#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "dqn_agent.h"
#include "cartpole.h"

int main(int argc, char **argv) {
  std::srand((unsigned)std::time(nullptr));

  torch::Device dispositivo = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Dispositivo: " << (dispositivo.is_cuda() ? "CUDA" : "CPU") << std::endl;

  CartPoleEnv entorno;
  const int dim_estado = 4;
  const int dim_accion = 2;
  DQNAgent agente(dim_estado, dim_accion, 128, 1e-3);
  agente.a_dispositivo(dispositivo);

  const int num_episodios = 500;
  const int frecuencia_actualizacion = 10;
  float epsilon = 1.0f;
  const float epsilon_final = 0.01f;
  const float tau_decay = 200.0f;
  int pasos_totales = 0;
  const int pasos_warmup = 1000;

  for (int episodio = 1; episodio <= num_episodios; ++episodio) {
    auto estado = entorno.reiniciar();
    float recompensa_episodio = 0.0f;
    
    for (int paso = 0; paso < 1000; ++paso) {
      auto tensor_estado = torch::from_blob(estado.data(), {4}, torch::kFloat).clone();
      int accion = agente.seleccionar_accion(tensor_estado, epsilon);
      auto [siguiente_estado, recompensa, terminado] = entorno.ejecutar(accion);
      Transicion transicion{estado, accion, recompensa, siguiente_estado, terminado};
      agente.recordar(transicion);
      
      if (pasos_totales >= pasos_warmup) {
        agente.optimizar();
      }
      
      estado = siguiente_estado;
      recompensa_episodio += recompensa;
      pasos_totales++;
      if (terminado) break;
    }

    epsilon = epsilon_final + (1.0f - epsilon_final) * std::exp(-1.0f * episodio / tau_decay);

    if (episodio % frecuencia_actualizacion == 0) agente.actualizar_red_objetivo();

    if (episodio % 10 == 0) {
      std::cout << "Episodio " << episodio << " | Recompensa: " << recompensa_episodio 
                << " | Epsilon: " << epsilon 
                << " | Pasos: " << pasos_totales << std::endl;
    }
    
    if (episodio % 50 == 0) {
      agente.guardar("checkpoint_ep" + std::to_string(episodio) + ".pt");
      std::cout << "Checkpoint guardado en episodio " << episodio << std::endl;
    }
  }

  std::cout << "Entrenamiento finalizado." << std::endl;
  agente.guardar("modelo_final.pt");
  std::cout << "Modelo final guardado en modelo_final.pt" << std::endl;
  return 0;
}
