#include "cartpole.h"
#include <cmath>
#include <random>
#include <tuple>

CartPoleEnv::CartPoleEnv()
    : gravedad_(9.8), masa_carro_(1.0), masa_poste_(0.1), longitud_poste_(0.5), 
      magnitud_fuerza_(10.0), intervalo_tiempo_(0.02) {
  masa_total_ = masa_poste_ + masa_carro_;
  masa_por_longitud_ = masa_poste_ * longitud_poste_;
  umbral_angulo_ = 12 * 2 * M_PI / 360.0;  // 12 grados en radianes
  umbral_posicion_ = 2.4f;
}

std::vector<float> CartPoleEnv::reiniciar() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
  estado_[0] = dist(gen);
  estado_[1] = dist(gen);
  estado_[2] = dist(gen);
  estado_[3] = dist(gen);
  return {estado_[0], estado_[1], estado_[2], estado_[3]};
}

std::tuple<std::vector<float>, float, bool> CartPoleEnv::ejecutar(int accion) {
  float x = estado_[0];
  float velocidad_x = estado_[1];
  float theta = estado_[2];
  float velocidad_angular = estado_[3];

  float fuerza = (accion == 1) ? magnitud_fuerza_ : -magnitud_fuerza_;
  float cos_theta = std::cos(theta);
  float sin_theta = std::sin(theta);

  float temp = (fuerza + masa_por_longitud_ * velocidad_angular * velocidad_angular * sin_theta) / masa_total_;
  float aceleracion_angular = (gravedad_ * sin_theta - cos_theta * temp) / 
                              (longitud_poste_ * (4.0f/3.0f - masa_poste_ * cos_theta * cos_theta / masa_total_));
  float aceleracion_x = temp - masa_por_longitud_ * aceleracion_angular * cos_theta / masa_total_;

  // Integración de Euler
  x += intervalo_tiempo_ * velocidad_x;
  velocidad_x += intervalo_tiempo_ * aceleracion_x;
  theta += intervalo_tiempo_ * velocidad_angular;
  velocidad_angular += intervalo_tiempo_ * aceleracion_angular;

  estado_[0] = x;
  estado_[1] = velocidad_x;
  estado_[2] = theta;
  estado_[3] = velocidad_angular;

  // Condiciones de terminación
  bool terminado = x < -umbral_posicion_ || x > umbral_posicion_ || 
                   theta < -umbral_angulo_ || theta > umbral_angulo_;
  float recompensa = terminado ? 0.0f : 1.0f;

  return {{estado_[0], estado_[1], estado_[2], estado_[3]}, recompensa, terminado};
}
