# Hyderabad Traffic Simulation using Cellular Automata

A sophisticated traffic simulation system that uses cellular automata principles to model realistic vehicular traffic flow on Hyderabad city's road network.

![Traffic Simulation Demo](final_traffic_simulation.png)

## 🎯 Project Overview

This project extracts road networks from satellite imagery and simulates traffic flow using cellular automata. The system includes:

- **Road Network Extraction**: Converts black-and-white map data into a computational grid
- **Cellular Automata Engine**: Implements traffic rules for car movement and interactions
- **Traffic Management**: Strategic traffic light placement and timing
- **Real-time Visualization**: Live animation overlaid on the original city map
- **Performance Analytics**: Comprehensive metrics and congestion analysis

## 🚀 Features

### Core Simulation Engine
- ✅ **4-directional car movement** (North, South, East, West)
- ✅ **Collision avoidance** - cars wait if next cell is occupied
- ✅ **Traffic signal compliance** - cars stop at red lights
- ✅ **Dynamic spawning** at road entry points
- ✅ **Intelligent direction changes** when blocked
- ✅ **Congestion detection** and adaptive behavior

### Advanced Traffic Management
- 🚥 **Strategic traffic light placement** at intersections
- 🚗 **Adaptive car spawning** based on congestion levels
- 📊 **Real-time congestion monitoring**
- 🎯 **Traffic efficiency optimization**
- 📈 **Performance metrics tracking**

### Visualization & Analysis
- 🖼️ **High-quality overlay** on original Hyderabad map
- 🎨 **Color-coded cars** (green=moving, yellow=slow, red=stuck)
- 📊 **Live metrics display** (cars, congestion, efficiency)
- 🎬 **Animation frame export** for time-lapse analysis
- 📄 **Comprehensive reporting** with detailed statistics

## 📁 Project Structure

```
├── hyd_bw.png                    # Black & white road map (input)
├── hyd.png                       # Color Hyderabad map (background)
├── traffic_simulation.py         # Full pygame-based simulation
├── final_traffic_simulation.py   # Optimized final version ⭐
├── enhanced_traffic_sim.py       # Advanced features version
├── headless_simulation.py        # Console-based simulation
├── requirements.txt              # Python dependencies
├── final_traffic_simulation.png  # Final visualization output
├── final_traffic_step_*.png      # Animation frames
└── final_traffic_analysis.txt    # Comprehensive metrics report
```

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Cellular-Automata---hyderabad-traffic-simulation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify input images**:
   - Ensure `hyd_bw.png` (road network) and `hyd.png` (city map) are present
   - Both images should have the same dimensions

## 🚀 Usage

### Run the Complete Simulation
```bash
python3 final_traffic_simulation.py
```

### Alternative Versions
```bash
# Interactive pygame version (requires display)
python3 traffic_simulation.py

# Headless version (no GUI)
python3 headless_simulation.py

# Enhanced features version
python3 enhanced_traffic_sim.py
```

## 📊 Simulation Results

### Latest Performance Metrics
- **🚗 Total Cars Processed**: 183
- **✅ Journey Completion Rate**: 63.9%
- **📈 Traffic Efficiency**: 0.64
- **🚧 Average Congestion**: 60.7%
- **⏱️ Simulation Runtime**: 1.12 seconds (400 steps)

### Key Insights
- **Peak Traffic**: 69 cars simultaneously active
- **Congestion Patterns**: 61% average congestion with peaks at 84%
- **Traffic Flow**: Effective distribution across 33 entry points
- **Infrastructure**: 50 traffic lights managing 6,039 road cells

## 🎮 Interactive Controls

For pygame-based versions:
- **ESC**: Exit simulation
- **SPACE**: Pause/Resume
- **Mouse**: View real-time metrics

## 📈 Output Files

The simulation generates several output files:

1. **`final_traffic_simulation.png`** - Complete visualization with overlay
2. **`final_traffic_step_*.png`** - Animation frames at key intervals
3. **`final_traffic_analysis.txt`** - Comprehensive performance report
4. **Traffic metrics** - Step-by-step congestion and efficiency data

## 🧮 Cellular Automata Rules

### Car Movement Rules
1. **Forward Movement**: Move to next cell if it's a road and unoccupied
2. **Traffic Signal Compliance**: Stop at red lights
3. **Collision Avoidance**: Wait if next cell contains another car
4. **Direction Changes**: Change direction when stuck for too long
5. **Exit Behavior**: Remove cars that reach simulation boundaries

### Traffic Light Logic
- **Red Phase**: 20 time steps (cars must stop)
- **Green Phase**: 30 time steps (cars can proceed)
- **Strategic Placement**: At intersections with high connectivity

### Spawning Algorithm
- **Entry Points**: Road cells at simulation boundaries
- **Adaptive Rate**: Spawn rate decreases with congestion
- **Capacity Limits**: Maximum 150 cars simultaneously
- **Direction Assignment**: Based on entry point location

## 🔬 Technical Implementation

### Road Network Extraction
```python
# Extract roads from black pixels in hyd_bw.png
road_threshold = 80
road_grid = resized_image < road_threshold
```

### Cellular Automata Core
```python
# Each cell can be: EMPTY, ROAD, CAR, or TRAFFIC_LIGHT
# Cars move according to direction vectors:
directions = {
    NORTH: (0, -1), SOUTH: (0, 1),
    EAST: (1, 0), WEST: (-1, 0)
}
```

### Performance Optimization
- **Efficient collision detection** using position hashing
- **Randomized car processing** to prevent movement bias
- **Adaptive spawning** based on real-time congestion
- **Optimized visualization** with selective rendering

## 📊 Algorithm Complexity

- **Time Complexity**: O(n) per step, where n = number of cars
- **Space Complexity**: O(w×h) for grid storage
- **Scalability**: Handles up to 200+ concurrent cars efficiently
- **Frame Rate**: ~10 FPS for real-time visualization

## 🌟 Advanced Features

### Congestion Management
- **Dynamic spawn rate adjustment**
- **Intelligent direction changing for stuck cars**
- **Traffic light timing optimization**
- **Bottleneck detection and analysis**

### Visualization Enhancements
- **Status-based car coloring**
- **Direction indicators with arrows**
- **Traffic light glow effects**
- **Real-time metrics overlay**
- **Performance analytics dashboard**

## 🔮 Future Enhancements

- [ ] **Multi-lane roads** with lane-changing behavior
- [ ] **Different vehicle types** (cars, buses, motorcycles)
- [ ] **Rush hour simulation** with time-based spawning
- [ ] **Route optimization** and GPS-like pathfinding
- [ ] **Accident simulation** and emergency vehicle priority
- [ ] **Machine learning** for adaptive traffic light timing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for image processing capabilities
- **NumPy** for efficient array operations
- **Pygame** for real-time visualization
- **Matplotlib** for data visualization and analysis
- **Cellular Automata principles** from computational physics research

---

*Built with ❤️ for traffic simulation and urban planning research*
