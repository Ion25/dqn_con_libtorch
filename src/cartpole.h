#pragma once

#include <array>
#include <vector>

// Entorno CartPole: balancear un poste sobre un carro móvil
class CartPoleEnv {
public:
  CartPoleEnv();
  std::vector<float> reiniciar();
  
  // Ejecuta acción: 0 (izquierda), 1 (derecha)
  // Retorna: (siguiente_estado, recompensa, terminado)
  std::tuple<std::vector<float>, float, bool> ejecutar(int accion);

private:
  // Estado: x, velocidad_x, theta, velocidad_angular
  std::array<float,4> estado_;
  float gravedad_;
  float masa_carro_;
  float masa_poste_;
  float masa_total_;
  float longitud_poste_;
  float masa_por_longitud_;
  float magnitud_fuerza_;
  float intervalo_tiempo_;  // Intervalo de actualización
  float umbral_angulo_;
  float umbral_posicion_;
};
