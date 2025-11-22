# Deep Q-Network (DQN) en C++ con LibTorch

Implementación completa de Deep Q-Network para aprendizaje por refuerzo en C++17 utilizando LibTorch. El proyecto incluye optimizaciones modernas como Double DQN, gradient clipping, experience replay y soporte para aceleración GPU mediante CUDA.

## Descripción del Proyecto

Este proyecto implementa un agente de aprendizaje por refuerzo basado en DQN que aprende a resolver el problema clásico de control CartPole. La implementación está completamente en C++ para obtener máximo rendimiento y facilitar el despliegue en producción.

### Características Principales

- Deep Q-Network con arquitectura MLP (3 capas)
- Double DQN para reducir sobreestimación de valores Q
- Experience Replay con buffer circular (100,000 transiciones)
- Target Network con actualizaciones periódicas
- Epsilon-greedy con decay exponencial
- Gradient clipping para estabilidad del entrenamiento
- Warm-up del replay buffer
- Checkpointing automático cada 50 episodios
- Detección automática de GPU/CPU
- Código completamente en español (variables y comentarios)

### Algoritmo

El agente utiliza el algoritmo DQN propuesto por Mnih et al. (2015), con las siguientes mejoras:

**Ecuación de Bellman (DQN estándar):**
```
Q(s,a) = r + γ max Q(s',a')
         a'
```

**Double DQN:**
```
Q(s,a) = r + γ Q(s', argmax Q(s',a'; θ); θ⁻)
                    a'
```

Donde:
- `θ`: parámetros de la red policy (se entrena)
- `θ⁻`: parámetros de la red objetivo (actualización periódica)
- `γ`: factor de descuento (0.99)

## Estructura del Proyecto

```
.
├── CMakeLists.txt                  # Configuración de CMake
├── README.md                       # Este archivo
├── resultados_entrenamiento.txt    # Log de ejemplo del entrenamiento
└── src/
    ├── main.cpp                   # Loop principal de entrenamiento
    ├── model.h                    # Arquitectura de la red neuronal
    ├── dqn_agent.h/.cpp          # Agente DQN completo
    ├── replay_buffer.h/.cpp      # Buffer de experiencias
    └── cartpole.h/.cpp           # Entorno de simulación
```

## Requisitos del Sistema

### Software Necesario

- **CMake** >= 3.10
- **Compilador C++17**:
  - GCC >= 7.0
  - Clang >= 5.0
  - MSVC >= 2017
- **LibTorch** 2.1.0 o superior
- **CUDA Toolkit** (opcional, solo para GPU)

### Hardware Recomendado

- **CPU**: Cualquier procesador moderno (Intel/AMD)
- **RAM**: Mínimo 4 GB
- **GPU** (opcional): NVIDIA con soporte CUDA 11.8+

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/DQN-LibTorch.git
cd DQN-LibTorch
```

### 2. Descargar LibTorch

#### Opción A: Script Automático (Linux)

```bash
# Para CPU-only
./download_libtorch.sh cpu

# Para CUDA 11.8
./download_libtorch.sh cu118

# Para CUDA 12.1
./download_libtorch.sh cu121
```

#### Opción B: Descarga Manual

1. Visita https://pytorch.org/get-started/locally/
2. Selecciona:
   - Package: **LibTorch**
   - Language: **C++/Java**
   - Compute Platform: **CPU** o tu versión de CUDA
   - Download and install: Copia el enlace
3. Descarga y descomprime en el directorio del proyecto

```bash
wget [URL_DE_LIBTORCH]
unzip libtorch-*.zip
```

### 3. Compilar el Proyecto

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
make -j$(nproc)
```

Si LibTorch está en otra ubicación:

```bash
cmake -DCMAKE_PREFIX_PATH=/ruta/a/libtorch ..
```

## Ejecución

### Entrenar el Agente

```bash
cd build
./dqn
```

### Guardar Resultados del Entrenamiento

```bash
./dqn 2>&1 | tee resultados_entrenamiento.txt
```

### Salida Esperada

```
Dispositivo: CPU
Episodio 10 | Recompensa: 22 | Epsilon: 0.951717 | Pasos: 184
Episodio 20 | Recompensa: 25 | Epsilon: 0.905789 | Pasos: 431
Episodio 30 | Recompensa: 21 | Epsilon: 0.862101 | Pasos: 637
...
Episodio 500 | Recompensa: 438 | Epsilon: 0.091264 | Pasos: 47167
Checkpoint guardado en episodio 500
Entrenamiento finalizado.
Modelo final guardado en modelo_final.pt
```

<img width="676" height="230" alt="resultado_de_entrenamiento" src="https://github.com/user-attachments/assets/51fe96d5-7fbc-40dc-b361-8f90558520d0" />



## Configuración y Parámetros

Los hiperparámetros se pueden modificar en `src/main.cpp`:

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `num_episodios` | 500 | Número total de episodios |
| `dim_estado` | 4 | Dimensión del espacio de estados |
| `dim_accion` | 2 | Número de acciones posibles |
| `neuronas_ocultas` | 128 | Neuronas en capas ocultas |
| `tasa_aprendizaje` | 1e-3 | Learning rate (Adam) |
| `tamano_lote` | 64 | Tamaño del minibatch |
| `gamma` | 0.99 | Factor de descuento |
| `epsilon_inicial` | 1.0 | Exploración inicial |
| `epsilon_final` | 0.01 | Exploración mínima |
| `tau_decay` | 200.0 | Velocidad de decay exponencial |
| `pasos_warmup` | 1000 | Pasos antes de entrenar |
| `frecuencia_actualizacion` | 10 | Actualizar red objetivo cada N episodios |

## Arquitectura de la Red Neuronal

```
Entrada (4) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2) → Salida
```

- **Entrada**: Estado del entorno (posición, velocidad, ángulo, velocidad angular)
- **Salida**: Valores Q para cada acción (izquierda, derecha)

## Resultados

Ver `resultados_entrenamiento.txt` para un ejemplo completo de un entrenamiento de 500 episodios.

### Métricas de Rendimiento

- **Episodios iniciales (1-50)**: Recompensa promedio ~10-25
- **Episodios intermedios (100-250)**: Recompensa promedio ~50-130
- **Episodios finales (400-500)**: Recompensa promedio ~100-400
- **Mejor episodio**: 709 pasos (episodio 360)

### Checkpoints Generados

Los checkpoints se guardan automáticamente en el directorio `build/`:

- `checkpoint_ep50.pt`
- `checkpoint_ep100.pt`
- ...
- `checkpoint_ep500.pt`
- `modelo_final.pt`

## Entorno de Prueba: CartPole

CartPole es un problema clásico de control donde un poste debe mantenerse vertical sobre un carro que se mueve horizontalmente.

**Estado (4 dimensiones):**
- Posición del carro: x
- Velocidad del carro: ẋ
- Ángulo del poste: θ
- Velocidad angular: θ̇

**Acciones (2):**
- 0: Mover carro a la izquierda
- 1: Mover carro a la derecha

**Recompensa:**
- +1 por cada paso que el poste permanece en equilibrio
- 0 si el episodio termina

**Terminación:**
- |x| > 2.4 (carro fuera de límites)
- |θ| > 12° (poste cayó)

## Documentación Adicional

### Documento Técnico LaTeX

El proyecto incluye documentación técnica completa en LaTeX:

```bash
cd doc
make
```

Esto genera `proyecto_dqn.pdf` con:
- Fundamentos teóricos
- Arquitectura detallada
- Pseudocódigo del algoritmo
- Análisis de resultados
- Referencias bibliográficas

## Solución de Problemas

### Error: "Could not find Torch"

Verifica que la ruta a LibTorch sea correcta:

```bash
cmake -DCMAKE_PREFIX_PATH=/ruta/correcta/a/libtorch ..
```

### Error de CUDA en Runtime

Asegúrate de que la versión de CUDA de LibTorch coincida con tu CUDA Toolkit:

```bash
nvcc --version  # Ver versión instalada
```

Descarga la build de LibTorch correspondiente (cu118, cu121, etc.)

### Compilación Lenta

Usa múltiples cores:

```bash
make -j$(nproc)  # Linux/Mac
make -j%NUMBER_OF_PROCESSORS%  # Windows
```

## Trabajo Futuro

Mejoras potenciales al proyecto:

- [ ] Prioritized Experience Replay
- [ ] Dueling DQN architecture
- [ ] Noisy Networks para exploración
- [ ] N-step returns
- [ ] Rainbow DQN (combinación de mejoras)
- [ ] Integración con OpenAI Gym vía pybind11
- [ ] Soporte para entornos Atari
- [ ] Logging con TensorBoard
- [ ] Entrenamiento distribuido

## Referencias

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
2. Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double q-learning." AAAI.
3. Schaul, T., et al. (2015). "Prioritized experience replay." arXiv:1511.05952.
4. LibTorch Documentation: https://pytorch.org/cppdocs/

