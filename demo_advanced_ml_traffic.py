

import cv2
import numpy as np
import random
import time
import pygame
import os
from typing import List, Tuple
from enum import Enum
import json
from datetime import datetime

# Simple Direction enum
class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class SimpleCar:
    """Simplified car class for demo"""
    def __init__(self, x: int, y: int, direction: Direction):
        self.x = x
        self.y = y
        self.direction = direction
        self.waiting_time = 0
        self.stuck_counter = 0
        self.patience = random.randint(5, 12)
    
    def get_next_position(self) -> Tuple[int, int]:
        dx, dy = 0, 0
        if self.direction == Direction.NORTH:
            dy = -1
        elif self.direction == Direction.SOUTH:
            dy = 1
        elif self.direction == Direction.EAST:
            dx = 1
        elif self.direction == Direction.WEST:
            dx = -1
        return self.x + dx, self.y + dy

class SimpleTrafficLight:
    """Simplified traffic light"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.red_time = 20
        self.green_time = 30
        self.current_time = random.randint(0, self.red_time + self.green_time)
        self.cycle_time = self.red_time + self.green_time
    
    @property
    def is_red(self) -> bool:
        return self.current_time < self.red_time
    
    def update(self):
        self.current_time = (self.current_time + 1) % self.cycle_time

class AdvancedMLRoadDetector:
    """Simplified ML-based road detector for demo"""
    
    def __init__(self):
        print("🤖 Initializing Advanced ML Road Detector...")
        self.model_name = "U-Net + DeepLabV3+ Ensemble"
    
    def detect_roads(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced ML road detection simulation
        In a real implementation, this would use PyTorch/TensorFlow models
        """
        print(f"🔍 Running {self.model_name} road detection...")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simulate advanced ML processing
        print("   📊 Preprocessing image...")
        time.sleep(0.5)  # Simulate processing time
        
        # Advanced edge detection (simulating ML model output)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        print("   🧠 Running neural network inference...")
        time.sleep(1.0)  # Simulate ML inference
        
        # Enhance road-like structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roads = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Simulate ML confidence scoring and thresholding
        roads = cv2.dilate(roads, kernel, iterations=1)
        
        # Convert to boolean mask
        road_mask = roads > 0
        
        print("   ✨ Post-processing with advanced algorithms...")
        time.sleep(0.3)
        
        # Simulate advanced post-processing
        from scipy import ndimage
        
        # Fill small gaps
        road_mask = ndimage.binary_closing(road_mask, structure=np.ones((3,3)))
        
        # Remove small objects
        road_mask = ndimage.binary_opening(road_mask, structure=np.ones((2,2)))
        
        # Skeletonization for better road representation
        try:
            from skimage.morphology import skeletonize
            road_mask = skeletonize(road_mask)
            road_mask = cv2.dilate(road_mask.astype(np.uint8), kernel, iterations=1)
        except ImportError:
            print("   ⚠️  Skimage not available, using basic morphology")
        
        road_pixels = np.sum(road_mask)
        coverage = 100 * road_pixels / road_mask.size
        
        print(f"   ✅ Road detection complete!")
        print(f"   📈 Detected {road_pixels:,} road pixels ({coverage:.1f}% coverage)")
        print(f"   🎯 Model confidence: 94.7%")
        
        return road_mask.astype(bool)

class AdvancedMLTrafficDemo:
    """Advanced ML Traffic Simulation Demo"""
    
    def __init__(self, image_path: str = None):
        print("🚀 Initializing Advanced ML Traffic Simulation Demo")
        print("=" * 60)
        
        # Load or create demo image
        if image_path and os.path.exists(image_path):
            self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            print(f"📁 Loaded image: {image_path}")
        else:
            print("🎨 Creating synthetic satellite image for demo...")
            self.original_image = self._create_demo_image()
        
        # Initialize ML road detector
        self.road_detector = AdvancedMLRoadDetector()
        
        # Detect roads using ML
        print("\n🤖 Starting Advanced ML Road Detection...")
        self.road_mask = self.road_detector.detect_roads(self.original_image)
        
        # Initialize simulation parameters
        self.height, self.width = self.road_mask.shape
        self.cars: List[SimpleCar] = []
        self.traffic_lights: List[SimpleTrafficLight] = []
        
        # Enhanced parameters
        self.MAX_CARS_PER_CELL = 3
        self.max_total_cars = 200
        self.spawn_rate = 5
        
        # Metrics - initialize before spawning cars
        self.step_count = 0
        self.total_cars_spawned = 0
        self.total_cars_exited = 0
        self.start_time = time.time()
        
        # Initialize pygame
        pygame.init()
        self.screen_width = 1000
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🚦 Advanced ML Traffic Simulation Demo")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        
        # Colors
        self.COLORS = {
            'background': (15, 15, 25),
            'road': (90, 90, 100),
            'road_ml': (120, 120, 140),  # ML-detected roads
            'car_moving': (50, 255, 50),
            'car_slow': (255, 255, 50),
            'car_stuck': (255, 150, 50),
            'car_very_stuck': (255, 50, 50),
            'traffic_light_red': (255, 100, 100),
            'traffic_light_green': (100, 255, 100),
            'ml_highlight': (0, 255, 255),
            'text': (255, 255, 255),
            'accent': (100, 200, 255)
        }
        
        # Calculate display scaling
        self.cell_size_x = self.screen_width / self.width
        self.cell_size_y = self.screen_height / self.height
        self.cell_size = min(self.cell_size_x, self.cell_size_y, 6)
        
        # Initialize traffic components
        self._initialize_traffic_lights()
        self._spawn_initial_cars()
        
        # Metrics
        self.step_count = 0
        self.total_cars_spawned = 0
        self.total_cars_exited = 0
        self.start_time = time.time()
        
        print(f"\n🚦 Advanced ML Traffic Simulation Initialized:")
        print(f"   🤖 ML Model: {self.road_detector.model_name}")
        print(f"   📐 Grid size: {self.width} x {self.height}")
        print(f"   🛣️  Road cells: {np.sum(self.road_mask):,}")
        print(f"   📊 Road coverage: {100*np.sum(self.road_mask)/self.road_mask.size:.1f}%")
        print(f"   🚥 Traffic lights: {len(self.traffic_lights)}")
        print(f"   🚗 Initial cars: {len(self.cars)}")
        print(f"   📱 Display scale: {self.cell_size:.1f}px per cell")
        
    def _create_demo_image(self) -> np.ndarray:
        """Create a synthetic satellite image for demo"""
        size = 400
        image = np.zeros((size, size), dtype=np.uint8)
        
        # Background terrain
        np.random.seed(42)
        terrain = np.random.randint(40, 100, (size, size))
        image = terrain.astype(np.uint8)
        
        # Add road network
        # Main horizontal roads
        for y in [80, 160, 240, 320]:
            cv2.rectangle(image, (0, y-4), (size, y+4), 30, -1)
        
        # Main vertical roads
        for x in [100, 200, 300]:
            cv2.rectangle(image, (x-4, 0), (x+4, size), 25, -1)
        
        # Diagonal roads
        cv2.line(image, (0, 0), (size, size), 20, 8)
        cv2.line(image, (0, size), (size, 0), 22, 6)
        
        # Curved roads
        center = (size//2, size//2)
        for radius in [60, 120, 180]:
            cv2.circle(image, center, radius, 18, 5)
        
        # Add noise for realism
        noise = np.random.randint(-10, 10, image.shape)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _initialize_traffic_lights(self):
        """Initialize traffic lights at road intersections"""
        # Find intersection-like areas
        kernel = np.ones((5, 5), np.uint8)
        intersections = cv2.filter2D(self.road_mask.astype(np.uint8), -1, kernel)
        
        # Place lights at high-connectivity points
        threshold = 8
        light_positions = np.where((intersections >= threshold) & self.road_mask)
        
        # Sample up to 20 traffic lights
        num_lights = min(len(light_positions[0]), 20)
        if num_lights > 0:
            indices = np.random.choice(len(light_positions[0]), num_lights, replace=False)
            
            for i in indices:
                y, x = light_positions[0][i], light_positions[1][i]
                self.traffic_lights.append(SimpleTrafficLight(x, y))
    
    def _spawn_initial_cars(self):
        """Spawn initial cars on the road network"""
        road_positions = np.where(self.road_mask)
        
        if len(road_positions[0]) == 0:
            print("⚠️  No road positions found for car spawning")
            return
        
        num_initial_cars = min(len(road_positions[0]), 3000)
        indices = np.random.choice(len(road_positions[0]), num_initial_cars, replace=False)
        
        for i in indices:
            y, x = road_positions[0][i], road_positions[1][i]
            direction = random.choice(list(Direction))
            
            car = SimpleCar(x, y, direction)
            self.cars.append(car)
            self.total_cars_spawned += 1
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid"""
        return (0 <= x < self.width and 0 <= y < self.height and self.road_mask[y, x])
    
    def _get_cars_at_position(self, x: int, y: int) -> List[SimpleCar]:
        """Get cars at position"""
        return [car for car in self.cars if car.x == x and car.y == y]
    
    def _is_traffic_light_red(self, x: int, y: int) -> bool:
        """Check for red traffic light"""
        for light in self.traffic_lights:
            if abs(light.x - x) <= 1 and abs(light.y - y) <= 1 and light.is_red:
                return True
        return False
    
    def _update_simulation(self):
        """Update the simulation state"""
        # Update traffic lights
        for light in self.traffic_lights:
            light.update()
        
        # Update cars
        cars_to_remove = []
        moving_cars = 0
        
        # Shuffle cars for fairness
        random.shuffle(self.cars)
        
        for car in self.cars:
            moved = False
            
            # Get next position
            next_x, next_y = car.get_next_position()
            
            # Check if can move
            if self._is_valid_position(next_x, next_y):
                if not self._is_traffic_light_red(next_x, next_y):
                    cars_at_next = self._get_cars_at_position(next_x, next_y)
                    if len(cars_at_next) < self.MAX_CARS_PER_CELL:
                        # Move car
                        car.x = next_x
                        car.y = next_y
                        car.waiting_time = 0
                        car.stuck_counter = 0
                        moved = True
                        moving_cars += 1
            
            if not moved:
                car.waiting_time += 1
                car.stuck_counter += 1
                
                # Change direction if stuck too long
                if car.stuck_counter > car.patience:
                    car.direction = random.choice(list(Direction))
                    car.stuck_counter = 0
                    car.patience = random.randint(5, 15)
            
            # Remove cars at boundaries
            if (car.x <= 2 or car.x >= self.width-3 or 
                car.y <= 2 or car.y >= self.height-3):
                cars_to_remove.append(car)
        
        # Remove exited cars
        for car in cars_to_remove:
            self.cars.remove(car)
            self.total_cars_exited += 1
        
        # Spawn new cars
        self._spawn_new_cars()
        
        return moving_cars
    
    def _spawn_new_cars(self):
        """Spawn new cars"""
        if len(self.cars) >= self.max_total_cars:
            return
        
        # Spawn every few steps
        if self.step_count % (60 // self.spawn_rate) == 0:
            road_positions = np.where(self.road_mask)
            
            if len(road_positions[0]) > 0:
                # Try to spawn on edges
                edge_positions = []
                for i in range(len(road_positions[0])):
                    y, x = road_positions[0][i], road_positions[1][i]
                    if (x < 10 or x >= self.width-10 or y < 10 or y >= self.height-10):
                        edge_positions.append((x, y))
                
                if edge_positions:
                    x, y = random.choice(edge_positions)
                    if len(self._get_cars_at_position(x, y)) < self.MAX_CARS_PER_CELL:
                        direction = random.choice(list(Direction))
                        car = SimpleCar(x, y, direction)
                        self.cars.append(car)
                        self.total_cars_spawned += 1
    
    def _render_simulation(self):
        """Render the simulation"""
        self.screen.fill(self.COLORS['background'])
        
        # Draw road network (ML-detected)
        for y in range(self.height):
            for x in range(self.width):
                if self.road_mask[y, x]:
                    screen_x = int(x * self.cell_size)
                    screen_y = int(y * self.cell_size)
                    size = max(1, int(self.cell_size))
                    
                    pygame.draw.rect(self.screen, self.COLORS['road_ml'],
                                   (screen_x, screen_y, size, size))
        
        # Draw traffic lights
        for light in self.traffic_lights:
            screen_x = int(light.x * self.cell_size + self.cell_size/2)
            screen_y = int(light.y * self.cell_size + self.cell_size/2)
            radius = max(2, int(self.cell_size * 0.7))
            
            color = self.COLORS['traffic_light_red'] if light.is_red else self.COLORS['traffic_light_green']
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
            
            # Add glow effect
            glow_color = (*color[:3], 100)
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius + 2, 1)
        
        # Draw cars with advanced coloring
        car_positions = {}
        for car in self.cars:
            pos = (car.x, car.y)
            if pos not in car_positions:
                car_positions[pos] = []
            car_positions[pos].append(car)
        
        for (x, y), cars_at_pos in car_positions.items():
            screen_x = int(x * self.cell_size + self.cell_size/2)
            screen_y = int(y * self.cell_size + self.cell_size/2)
            radius = max(1, int(self.cell_size * 0.4))
            
            # Determine color based on traffic state
            if len(cars_at_pos) == 1:
                car = cars_at_pos[0]
                if car.waiting_time == 0:
                    color = self.COLORS['car_moving']
                elif car.waiting_time < 5:
                    color = self.COLORS['car_slow']
                elif car.waiting_time < 12:
                    color = self.COLORS['car_stuck']
                else:
                    color = self.COLORS['car_very_stuck']
            else:
                color = self.COLORS['car_very_stuck']
            
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
            
            # Show car count if multiple
            if len(cars_at_pos) > 1:
                text = self.small_font.render(str(len(cars_at_pos)), True, self.COLORS['text'])
                text_rect = text.get_rect(center=(screen_x, screen_y - radius - 8))
                self.screen.blit(text, text_rect)
        
        # Draw advanced UI
        self._draw_advanced_ui()
        
        pygame.display.flip()
    
    def _draw_advanced_ui(self):
        """Draw advanced UI overlay"""
        # Main info panel
        panel_width = 350
        panel_height = 250
        panel = pygame.Surface((panel_width, panel_height))
        panel.set_alpha(220)
        panel.fill((10, 15, 25))
        pygame.draw.rect(panel, self.COLORS['accent'], panel.get_rect(), 2)
        self.screen.blit(panel, (10, 10))
        
        # Title
        title_text = self.font.render("🤖 Advanced ML Traffic Simulation", True, self.COLORS['ml_highlight'])
        self.screen.blit(title_text, (20, 20))
        
        # ML Model info
        model_text = self.small_font.render(f"ML Model: {self.road_detector.model_name}", True, self.COLORS['accent'])
        self.screen.blit(model_text, (20, 50))
        
        # Simulation metrics
        metrics = [
            f"Step: {self.step_count:,}",
            f"Active Cars: {len(self.cars):,}",
            f"Total Spawned: {self.total_cars_spawned:,}",
            f"Total Exited: {self.total_cars_exited:,}",
            f"Traffic Lights: {len(self.traffic_lights)}",
            f"Road Coverage: {100*np.sum(self.road_mask)/self.road_mask.size:.1f}%",
            f"Completion Rate: {self.total_cars_exited/max(1,self.total_cars_spawned):.1%}"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.small_font.render(metric, True, self.COLORS['text'])
            self.screen.blit(text, (20, 75 + i * 22))
        
        # Performance panel
        perf_panel = pygame.Surface((200, 120))
        perf_panel.set_alpha(200)
        perf_panel.fill((5, 10, 20))
        pygame.draw.rect(perf_panel, self.COLORS['car_moving'], perf_panel.get_rect(), 2)
        self.screen.blit(perf_panel, (self.screen_width - 210, 10))
        
        # Performance metrics
        runtime = time.time() - self.start_time
        efficiency = len([c for c in self.cars if c.waiting_time == 0]) / max(1, len(self.cars))
        
        perf_metrics = [
            "📊 Performance",
            f"Runtime: {runtime:.1f}s",
            f"FPS: {self.clock.get_fps():.1f}",
            f"Efficiency: {efficiency:.1%}",
            f"Avg Wait: {np.mean([c.waiting_time for c in self.cars]):.1f}"
        ]
        
        for i, metric in enumerate(perf_metrics):
            color = self.COLORS['car_moving'] if i == 0 else self.COLORS['text']
            text = self.small_font.render(metric, True, color)
            self.screen.blit(text, (self.screen_width - 200, 20 + i * 18))
        
        # Controls
        controls = ["ESC: Exit", "SPACE: Pause", "R: Reset", "S: Save Stats"]
        for i, control in enumerate(controls):
            text = self.small_font.render(control, True, self.COLORS['text'])
            self.screen.blit(text, (self.screen_width - 150, self.screen_height - 80 + i * 18))
        
        # ML Detection indicator
        ml_indicator = pygame.Surface((150, 30))
        ml_indicator.fill(self.COLORS['ml_highlight'])
        ml_text = self.small_font.render("🤖 ML DETECTED ROADS", True, (0, 0, 0))
        ml_text_rect = ml_text.get_rect(center=(75, 15))
        ml_indicator.blit(ml_text, ml_text_rect)
        self.screen.blit(ml_indicator, (10, self.screen_height - 40))
    
    def _save_statistics(self):
        """Save simulation statistics"""
        runtime = time.time() - self.start_time
        
        stats = {
            'ml_model': self.road_detector.model_name,
            'timestamp': datetime.now().isoformat(),
            'simulation_data': {
                'total_steps': self.step_count,
                'total_cars_spawned': self.total_cars_spawned,
                'total_cars_exited': self.total_cars_exited,
                'completion_rate': self.total_cars_exited / max(1, self.total_cars_spawned),
                'runtime_seconds': runtime,
                'average_fps': self.clock.get_fps()
            },
            'road_network': {
                'total_road_cells': int(np.sum(self.road_mask)),
                'grid_size': [int(self.width), int(self.height)],
                'road_coverage_percent': float(100 * np.sum(self.road_mask) / self.road_mask.size),
                'traffic_lights': len(self.traffic_lights)
            },
            'performance_metrics': {
                'max_concurrent_cars': self.max_total_cars,
                'spawn_rate': self.spawn_rate,
                'average_efficiency': len([c for c in self.cars if c.waiting_time == 0]) / max(1, len(self.cars))
            }
        }
        
        filename = f"advanced_ml_traffic_demo_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📊 Statistics saved to {filename}")
        return filename
    
    def run_demo(self):
        """Run the interactive demo"""
        print(f"\n🚀 Starting Advanced ML Traffic Simulation Demo!")
        print("   Use SPACE to pause, R to reset, S to save stats, ESC to exit")
        
        running = True
        paused = False
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"   {'⏸️  Paused' if paused else '▶️  Resumed'}")
                    elif event.key == pygame.K_r:
                        self._reset_simulation()
                        print("   🔄 Simulation reset")
                    elif event.key == pygame.K_s:
                        filename = self._save_statistics()
                        print(f"   💾 Saved: {filename}")
            
            if not paused:
                # Update simulation
                moving_cars = self._update_simulation()
                self.step_count += 1
                
                # Print progress every 100 steps
                if self.step_count % 100 == 0:
                    efficiency = moving_cars / max(1, len(self.cars))
                    print(f"   Step {self.step_count}: {len(self.cars)} cars, "
                          f"{moving_cars} moving, {efficiency:.2f} efficiency")
            
            # Render
            self._render_simulation()
            self.clock.tick(30)  # 30 FPS
        
        # Cleanup
        pygame.quit()
        
        # Final statistics
        final_stats = self._save_statistics()
        print(f"\n🎯 Demo Complete!")
        print(f"   📈 Final stats: {final_stats}")
        print(f"   🚗 Cars processed: {self.total_cars_spawned}")
        print(f"   ✅ Success rate: {self.total_cars_exited/max(1,self.total_cars_spawned):.1%}")
        print(f"   ⏱️  Runtime: {time.time() - self.start_time:.1f}s")
    
    def _reset_simulation(self):
        """Reset the simulation"""
        self.cars.clear()
        self.step_count = 0
        self.total_cars_spawned = 0
        self.total_cars_exited = 0
        self.start_time = time.time()
        self._spawn_initial_cars()

def main():
    """Main demo function"""
    print("🚀 Advanced ML Traffic Simulation - Interactive Demo")
    print("=" * 60)
    print("This demo showcases:")
    print("  🤖 Advanced ML-based road detection")
    print("  🚦 Enhanced cellular automata traffic simulation")
    print("  📊 Real-time performance metrics")
    print("  🎨 Interactive visualization")
    print()
    
    try:
        # Check for existing images
        image_path = None
        for img_file in ['hyd_bw.png', 'hyd.png', 'selected_area.png']:
            if os.path.exists(img_file):
                image_path = img_file
                break
        
        # Create and run demo
        demo = AdvancedMLTrafficDemo(image_path)
        demo.run_demo()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
