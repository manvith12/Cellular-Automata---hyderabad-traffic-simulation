"""
Final Optimized Hyderabad Traffic Simulation

This version includes optimized parameters for realistic traffic simulation:
- Better road extraction with optimal scale factor
- Guaranteed car spawning with multiple entry strategies
- Enhanced visualization with detailed traffic patterns
- Comprehensive metrics and analysis
"""

import cv2
import numpy as np
import random
import time
import pygame
from enum import Enum
from typing import List, Tuple, Dict
import os

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class TrafficLight:
    def __init__(self, x: int, y: int, red_duration: int = 20, green_duration: int = 30):
        self.x = x
        self.y = y
        self.red_duration = red_duration
        self.green_duration = green_duration
        self.current_time = random.randint(0, red_duration + green_duration)
        self.cycle_time = red_duration + green_duration
        
    @property
    def is_red(self) -> bool:
        return self.current_time < self.red_duration
    
    def update(self):
        self.current_time = (self.current_time + 1) % self.cycle_time

class Car:
    def __init__(self, x: int, y: int, direction: Direction, car_id: int = None):
        self.x = x
        self.y = y
        self.direction = direction
        self.waiting_time = 0
        self.stuck_counter = 0
        self.patience = random.randint(5, 12)
        self.total_distance = 0
        self.car_id = car_id if car_id is not None else random.randint(1000, 9999)
        
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

class OptimizedTrafficSimulation:
    def __init__(self, bw_image_path: str, color_image_path: str):
        # Load images
        self.bw_image = cv2.imread(bw_image_path, cv2.IMREAD_GRAYSCALE)
        self.color_image = cv2.imread(color_image_path)
        
        if self.bw_image is None or self.color_image is None:
            raise FileNotFoundError("Could not load input images")
        
        # Optimized scale factor for better road extraction
        self.scale_factor = 3
        self.original_height, self.original_width = self.bw_image.shape
        
        # Extract road network with optimized parameters
        self.road_grid = self._extract_road_network()
        self.height, self.width = self.road_grid.shape
        
        # Multi-car cellular automata system
        self.MAX_CARS_PER_CELL = 30
        self.CONGESTION_THRESHOLD = 30
        self.cell_car_count = np.zeros((self.height, self.width), dtype=int)
        self.cars: List[Car] = []
        self.traffic_lights: List[TrafficLight] = []
        self.max_total_cars = 2000  # Increased for multi-car cells
        self.spawn_cooldown = 0
        self.force_spawn_counter = 0
        self.next_car_id = 1
        
        # Pygame initialization
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Hyderabad Traffic Simulation - Multi-Car Cellular Automata")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.running = True
        self.paused = False
        
        # Colors
        self.COLORS = {
            'background': (30, 30, 30),
            'road': (80, 80, 80),
            'car_moving': (0, 255, 0),
            'car_slow': (255, 255, 0),
            'car_stuck': (255, 100, 0),
            'car_very_stuck': (255, 0, 0),
            'traffic_light_red': (255, 0, 0),
            'traffic_light_green': (0, 255, 0),
            'congestion': (128, 0, 128),  # Purple for congested cells
            'text': (255, 255, 255),
            'overlay_bg': (0, 0, 0, 128)
        }
        
        # Scaling for display
        self.display_scale_x = self.screen_width / self.width
        self.display_scale_y = self.screen_height / self.height
        self.cell_size = min(self.display_scale_x, self.display_scale_y)
        
        # Metrics
        self.step_count = 0
        self.total_cars_spawned = 0
        self.total_cars_exited = 0
        self.metrics_history = []
        
        # Spawning control
        self.last_spawn_time = time.time()
        self.cars_per_second = 30
        self.initial_spawn_complete = False
        
        # Entry points cache
        self.entry_points = []
        
        self._initialize_traffic_lights()
        self._cache_entry_points()
        
        # Spawn 300 cars immediately at startup
        self._spawn_initial_cars(300)
        
        print(f"🚦 Multi-Car Traffic Simulation Initialized:")
        print(f"   📐 Original image: {self.original_width}x{self.original_height}")
        print(f"   🔲 Simulation grid: {self.width}x{self.height}")
        print(f"   🛣️  Road cells: {np.sum(self.road_grid)} ({100*np.sum(self.road_grid)/(self.width*self.height):.1f}%)")
        print(f"   🚥 Traffic lights: {len(self.traffic_lights)} (concentrated at bottom)")
        print(f"   🌐 Random spawn points: {len(self.entry_points)} (all road cells)")
        print(f"   📊 Max cars per cell: {self.MAX_CARS_PER_CELL}")
        print(f"   📊 Max total cars: {self.max_total_cars}")
        print(f"   🚗 Initial spawn: 300 cars immediately")
        print(f"   ⏰ Continuous spawn: {self.cars_per_second} cars/second")
        print(f"   🖥️  Display: {self.screen_width}x{self.screen_height} (reduced size)")
        print(f"   📱 Cell size: {self.cell_size:.1f}px")
    
    def _extract_road_network(self) -> np.ndarray:
        """Optimized road extraction"""
        # Resize with optimal scale
        new_height = self.original_height // self.scale_factor
        new_width = self.original_width // self.scale_factor
        
        resized = cv2.resize(self.bw_image, (new_width, new_height), 
                           interpolation=cv2.INTER_AREA)
        
        # Optimized threshold for better road detection
        road_threshold = 80
        road_grid = resized < road_threshold
        
        # Minimal cleaning to preserve road connectivity
        kernel = np.ones((2, 2), np.uint8)
        road_grid = road_grid.astype(np.uint8)
        road_grid = cv2.morphologyEx(road_grid, cv2.MORPH_CLOSE, kernel)
        
        return road_grid.astype(bool)
    
    def _initialize_traffic_lights(self):
        """Strategic traffic light placement with emphasis on bottom area"""
        placed_lights = 0
        
        # First, place traffic lights at the bottom of the map
        bottom_lights = 0
        for y in range(self.height - 30, self.height - 2):  # Bottom 30 rows
            for x in range(2, self.width - 2, 4):  # Every 4th position
                if self.road_grid[y, x] and bottom_lights < 20:
                    # Check for intersection patterns
                    neighbors = np.sum(self.road_grid[y-1:y+2, x-1:x+2])
                    if neighbors >= 4:  # Lower threshold for bottom area
                        self.traffic_lights.append(TrafficLight(x, y))
                        placed_lights += 1
                        bottom_lights += 1
        
        # Then place remaining lights throughout the map
        for y in range(2, self.height - 2, 3):  # Sample every 3rd position
            for x in range(2, self.width - 2, 3):
                if self.road_grid[y, x] and placed_lights < 50:
                    # Check for intersection patterns
                    neighbors = np.sum(self.road_grid[y-1:y+2, x-1:x+2])
                    if neighbors >= 5:  # Intersection criterion
                        if random.random() < 0.25:  # Random placement
                            self.traffic_lights.append(TrafficLight(x, y))
                            placed_lights += 1
    
    def _cache_entry_points(self):
        """Cache all road cells for random spawning"""
        self.entry_points = []
        
        # All road cells can be spawn points for truly random distribution
        for y in range(self.height):
            for x in range(self.width):
                if self.road_grid[y, x]:
                    # Choose random direction based on available neighbors
                    valid_dirs = self._get_valid_directions_for_spawn(x, y)
                    if valid_dirs:
                        direction = random.choice(valid_dirs)
                        self.entry_points.append((x, y, direction))
        
        print(f"   🚪 Spawn points found: {len(self.entry_points)} (all road cells)")
    
    def _spawn_initial_cars(self, count: int):
        """Spawn initial cars at startup"""
        print(f"🚗 Spawning {count} initial cars...")
        spawned = 0
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loop
        
        while spawned < count and attempts < max_attempts:
            attempts += 1
            if self.entry_points:
                x, y, direction = random.choice(self.entry_points)
                
                # Check if cell can accommodate more cars
                if self.cell_car_count[y, x] < self.MAX_CARS_PER_CELL:
                    new_car = Car(x, y, direction, self.next_car_id)
                    self.cars.append(new_car)
                    self.cell_car_count[y, x] += 1
                    self.total_cars_spawned += 1
                    self.next_car_id += 1
                    spawned += 1
        
        print(f"✅ Successfully spawned {spawned} initial cars")
        self.initial_spawn_complete = True
    
    def _get_valid_directions_for_spawn(self, x: int, y: int) -> List[Direction]:
        """Get valid directions for spawning at a position"""
        valid_dirs = []
        for direction in Direction:
            dx, dy = 0, 0
            if direction == Direction.NORTH:
                dy = -1
            elif direction == Direction.SOUTH:
                dy = 1
            elif direction == Direction.EAST:
                dx = 1
            elif direction == Direction.WEST:
                dx = -1
            
            if self._is_valid_position(x + dx, y + dy):
                valid_dirs.append(direction)
        
        return valid_dirs if valid_dirs else [Direction.NORTH]  # Default direction
    
    def _spawn_cars(self):
        """Enhanced car spawning: 30 cars per second after initial spawn"""
        if len(self.cars) >= self.max_total_cars:
            return
        
        current_time = time.time()
        
        # Check if one second has passed since last spawn
        if current_time - self.last_spawn_time >= 1.0:
            print(f"🕐 Spawning {self.cars_per_second} cars (1 second interval)")
            spawned = 0
            attempts = 0
            max_attempts = self.cars_per_second * 10
            
            while spawned < self.cars_per_second and attempts < max_attempts and len(self.cars) < self.max_total_cars:
                attempts += 1
                if self.entry_points:
                    x, y, direction = random.choice(self.entry_points)
                    
                    # Check if cell can accommodate more cars
                    if self.cell_car_count[y, x] < self.MAX_CARS_PER_CELL:
                        new_car = Car(x, y, direction, self.next_car_id)
                        self.cars.append(new_car)
                        self.cell_car_count[y, x] += 1
                        self.total_cars_spawned += 1
                        self.next_car_id += 1
                        spawned += 1
            
            self.last_spawn_time = current_time
            if spawned > 0:
                print(f"✅ Spawned {spawned} new cars (Total: {len(self.cars)})")
            else:
                print(f"⚠️  Could not spawn cars - network may be congested")
    
    def _get_cars_at_position(self, x: int, y: int) -> List[Car]:
        """Get all cars at a specific position"""
        return [car for car in self.cars if car.x == x and car.y == y]
    
    def _is_cell_congested(self, x: int, y: int) -> bool:
        """Check if a cell is congested"""
        if not self._is_valid_position(x, y):
            return False
        return self.cell_car_count[y, x] >= self.CONGESTION_THRESHOLD
    
    def _can_move_to_cell(self, x: int, y: int) -> bool:
        """Check if a car can move to a cell"""
        if not self._is_valid_position(x, y):
            return False
        return self.cell_car_count[y, x] < self.MAX_CARS_PER_CELL
    
    def _is_valid_position(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and 0 <= y < self.height and 
                self.road_grid[y, x])
    
    def _is_traffic_light_red(self, x: int, y: int) -> bool:
        for light in self.traffic_lights:
            if light.x == x and light.y == y and light.is_red:
                return True
        return False
    
    def _get_valid_directions(self, x: int, y: int) -> List[Direction]:
        valid_dirs = []
        for direction in Direction:
            dx, dy = 0, 0
            if direction == Direction.NORTH:
                dy = -1
            elif direction == Direction.SOUTH:
                dy = 1
            elif direction == Direction.EAST:
                dx = 1
            elif direction == Direction.WEST:
                dx = -1
            
            if self._is_valid_position(x + dx, y + dy):
                valid_dirs.append(direction)
        
        return valid_dirs
    
    def _move_cars(self):
        """Enhanced car movement logic with multi-car cells"""
        cars_to_remove = []
        moving_count = 0
        
        # Update cell car counts
        self.cell_car_count.fill(0)
        for car in self.cars:
            if self._is_valid_position(car.x, car.y):
                self.cell_car_count[car.y, car.x] += 1
        
        # Randomize car order to prevent bias
        car_indices = list(range(len(self.cars)))
        random.shuffle(car_indices)
        
        for i in car_indices:
            if i >= len(self.cars):
                continue
                
            car = self.cars[i]
            next_x, next_y = car.get_next_position()
            can_move = False
            
            # Check if movement is possible
            if self._is_valid_position(next_x, next_y):
                if self._can_move_to_cell(next_x, next_y):
                    if not self._is_traffic_light_red(next_x, next_y):
                        can_move = True
            
            if can_move:
                # Update cell counts for movement
                if self._is_valid_position(car.x, car.y):
                    self.cell_car_count[car.y, car.x] -= 1
                
                # Move the car
                car.x, car.y = next_x, next_y
                car.waiting_time = 0
                car.stuck_counter = 0
                car.total_distance += 1
                moving_count += 1
                
                if self._is_valid_position(car.x, car.y):
                    self.cell_car_count[car.y, car.x] += 1
                
                # Remove cars that reach borders (exit simulation) or have traveled too far
                if (car.x <= 0 or car.x >= self.width-1 or 
                    car.y <= 0 or car.y >= self.height-1 or
                    car.total_distance > 100):  # Max journey length
                    cars_to_remove.append(i)
                    self.total_cars_exited += 1
                    if self._is_valid_position(car.x, car.y):
                        self.cell_car_count[car.y, car.x] -= 1
            else:
                # Car is blocked
                car.waiting_time += 1
                car.stuck_counter += 1
                
                # Try to change direction if stuck
                if car.stuck_counter > car.patience:
                    valid_dirs = self._get_valid_directions(car.x, car.y)
                    available_dirs = []
                    
                    # Check which directions have space
                    for direction in valid_dirs:
                        dx, dy = 0, 0
                        if direction == Direction.NORTH:
                            dy = -1
                        elif direction == Direction.SOUTH:
                            dy = 1
                        elif direction == Direction.EAST:
                            dx = 1
                        elif direction == Direction.WEST:
                            dx = -1
                        
                        test_x, test_y = car.x + dx, car.y + dy
                        if self._can_move_to_cell(test_x, test_y):
                            available_dirs.append(direction)
                    
                    if available_dirs:
                        # Prefer directions that avoid opposite
                        preferred_dirs = []
                        opposite_dir = None
                        
                        if car.direction == Direction.NORTH:
                            opposite_dir = Direction.SOUTH
                        elif car.direction == Direction.SOUTH:
                            opposite_dir = Direction.NORTH
                        elif car.direction == Direction.EAST:
                            opposite_dir = Direction.WEST
                        elif car.direction == Direction.WEST:
                            opposite_dir = Direction.EAST
                        
                        for direction in available_dirs:
                            if direction != opposite_dir:
                                preferred_dirs.append(direction)
                        
                        if preferred_dirs:
                            car.direction = random.choice(preferred_dirs)
                        else:
                            car.direction = random.choice(available_dirs)
                    
                    car.stuck_counter = 0
                    car.patience = random.randint(5, 12)
                
                # Remove cars that have been stuck too long (prevent infinite accumulation)
                if car.waiting_time > 50:  # Remove cars stuck for 50+ steps
                    cars_to_remove.append(i)
                    if self._is_valid_position(car.x, car.y):
                        self.cell_car_count[car.y, car.x] -= 1
        
        # Remove cars that exited
        for i in sorted(cars_to_remove, reverse=True):
            if i < len(self.cars):
                self.cars.pop(i)
        
        return moving_count
    
    def _calculate_congestion(self) -> float:
        """Calculate congestion based on multi-car cells"""
        if not self.cars:
            return 0.0
        
        # Count congested cells
        congested_cells = np.sum(self.cell_car_count >= self.CONGESTION_THRESHOLD)
        total_occupied_cells = np.sum(self.cell_car_count > 0)
        
        if total_occupied_cells == 0:
            return 0.0
        
        return congested_cells / total_occupied_cells
    
    def _update_metrics(self, moving_count: int):
        total_cars = len(self.cars)
        blocked_cars = total_cars - moving_count
        congestion = self._calculate_congestion()
        congested_cells = np.sum(self.cell_car_count >= self.CONGESTION_THRESHOLD)
        
        metrics = {
            'step': self.step_count,
            'total_cars': total_cars,
            'moving_cars': moving_count,
            'blocked_cars': blocked_cars,
            'congestion_level': congestion,
            'congested_cells': congested_cells,
            'cars_spawned': self.total_cars_spawned,
            'cars_exited': self.total_cars_exited,
            'efficiency': self.total_cars_exited / max(self.total_cars_spawned, 1)
        }
        
        self.metrics_history.append(metrics)
    
    def _render_simulation(self):
        """Render the simulation using pygame"""
        self.screen.fill(self.COLORS['background'])
        
        # Draw road network and traffic state
        for y in range(self.height):
            for x in range(self.width):
                screen_x = int(x * self.cell_size)
                screen_y = int(y * self.cell_size)
                cell_rect = pygame.Rect(screen_x, screen_y, int(self.cell_size), int(self.cell_size))
                
                if self.road_grid[y, x]:
                    # Determine cell color based on congestion
                    car_count = self.cell_car_count[y, x]
                    
                    if car_count >= self.CONGESTION_THRESHOLD:
                        # Purple for congested cells
                        pygame.draw.rect(self.screen, self.COLORS['congestion'], cell_rect)
                    elif car_count > 0:
                        # Gradient from gray to orange based on car density
                        intensity = min(car_count / self.MAX_CARS_PER_CELL, 1.0)
                        color_r = int(80 + (255 - 80) * intensity)
                        color_g = int(80 + (165 - 80) * intensity)
                        color_b = 80
                        pygame.draw.rect(self.screen, (color_r, color_g, color_b), cell_rect)
                    else:
                        # Normal road color
                        pygame.draw.rect(self.screen, self.COLORS['road'], cell_rect)
                    
                    # Draw cell border for visibility
                    pygame.draw.rect(self.screen, (60, 60, 60), cell_rect, 1)
                    
                    # Draw car count if there are cars
                    if car_count > 0:
                        text_color = (255, 255, 255) if car_count < self.CONGESTION_THRESHOLD else (255, 255, 0)
                        count_text = self.small_font.render(str(car_count), True, text_color)
                        text_rect = count_text.get_rect(center=cell_rect.center)
                        self.screen.blit(count_text, text_rect)
        
        # Draw traffic lights
        for light in self.traffic_lights:
            screen_x = int(light.x * self.cell_size + self.cell_size // 2)
            screen_y = int(light.y * self.cell_size + self.cell_size // 2)
            
            color = self.COLORS['traffic_light_red'] if light.is_red else self.COLORS['traffic_light_green']
            radius = max(3, int(self.cell_size // 4))
            
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), radius + 1, 2)
        
        # Draw metrics overlay
        self._draw_metrics_overlay()
        
        # Draw pause indicator
        if self.paused:
            pause_text = self.font.render("PAUSED - Press SPACE to continue", True, (255, 255, 0))
            self.screen.blit(pause_text, (self.screen_width - 350, 10))
    
    def _draw_metrics_overlay(self):
        """Draw real-time metrics on screen"""
        if not self.metrics_history:
            return
        
        latest = self.metrics_history[-1]
        
        # Create semi-transparent overlay
        overlay = pygame.Surface((400, 300))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        
        metrics_text = [
            f"Step: {latest['step']}",
            f"Total Cars: {latest['total_cars']}",
            f"Moving: {latest['moving_cars']}",
            f"Blocked: {latest['blocked_cars']}",
            f"Congested Cells: {latest['congested_cells']}",
            f"Congestion Level: {latest['congestion_level']:.2f}",
            f"Cars Spawned: {latest['cars_spawned']}",
            f"Cars Exited: {latest['cars_exited']}",
            f"Efficiency: {latest['efficiency']:.2f}",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "ESC - Exit",
            "R - Reset Simulation"
        ]
        
        for i, text in enumerate(metrics_text):
            if text:
                color = self.COLORS['text']
                if "Congested" in text and latest['congested_cells'] > 10:
                    color = (255, 100, 100)
                elif "Congestion Level" in text and latest['congestion_level'] > 0.7:
                    color = (255, 100, 100)
                
                rendered = self.small_font.render(text, True, color)
                overlay.blit(rendered, (10, 10 + i * 20))
        
        self.screen.blit(overlay, (10, 10))
    
    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("Simulation paused" if self.paused else "Simulation resumed")
                elif event.key == pygame.K_r:
                    self._reset_simulation()
                    print("Simulation reset")
    
    def _reset_simulation(self):
        """Reset the simulation to initial state"""
        self.cars.clear()
        self.cell_car_count.fill(0)
        self.step_count = 0
        self.total_cars_spawned = 0
        self.total_cars_exited = 0
        self.metrics_history.clear()
        self.force_spawn_counter = 0
        self.next_car_id = 1
        self.last_spawn_time = time.time()
        self.initial_spawn_complete = False
        
        # Reset traffic lights
        for light in self.traffic_lights:
            light.current_time = random.randint(0, light.cycle_time)
        
        # Respawn initial cars
        self._spawn_initial_cars(300)
    
    def step(self):
        """Execute one simulation step"""
        if self.paused:
            return
        
        # Update traffic lights
        for light in self.traffic_lights:
            light.update()
        
        # Spawn new cars
        self._spawn_cars()
        
        # Move cars
        moving_count = self._move_cars()
        
        # Update metrics
        self._update_metrics(moving_count)
        
        self.step_count += 1
    
    def run_pygame_simulation(self):
        """Run the real-time pygame simulation"""
        print(f"\n🚀 Starting Real-Time Multi-Car Traffic Simulation")
        print("=" * 60)
        print(f"🎮 Controls:")
        print(f"   SPACE - Pause/Resume")
        print(f"   R - Reset simulation")
        print(f"   ESC - Exit")
        print(f"\n🚦 Enhanced Features:")
        print(f"   📊 Up to {self.MAX_CARS_PER_CELL} cars per cell")
        print(f"   💜 Purple cells indicate congestion (≥{self.CONGESTION_THRESHOLD} cars)")
        print(f"   🌐 Random spawning across entire road network")
        print(f"   🚗 300 cars spawn immediately at startup")
        print(f"   ⏰ {self.cars_per_second} new cars added every second")
        print(f"   🚥 Extra traffic lights at bottom of map")
        print(f"   📈 Real-time metrics and performance tracking")
        print(f"   🎯 Distributed traffic flow management")
        print(f"   📐 Compact {self.screen_width}x{self.screen_height} window")
        print("\nStarting simulation...")
        
        start_time = time.time()
        frame_count = 0
        
        while self.running:
            # Handle events
            self._handle_events()
            
            # Update simulation
            self.step()
            
            # Render
            self._render_simulation()
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(15)  # 15 FPS for smooth visualization
            frame_count += 1
            
            # Print periodic updates
            if self.step_count % 100 == 0 and self.step_count > 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                if self.metrics_history:
                    latest = self.metrics_history[-1]
                    print(f"Step {self.step_count:4d} | "
                          f"Cars: {latest['total_cars']:4d} | "
                          f"Congested: {latest['congested_cells']:3d} cells | "
                          f"Efficiency: {latest['efficiency']:.2f} | "
                          f"FPS: {fps:.1f}")
        
        # Cleanup
        elapsed_time = time.time() - start_time
        self._save_final_metrics()
        
        print(f"\n✅ Simulation completed!")
        print(f"⏱️  Runtime: {elapsed_time:.2f} seconds")
        print(f"🎬 Total frames: {frame_count}")
        print(f"📊 Final statistics:")
        
        if self.metrics_history:
            final_metrics = self.metrics_history[-1]
            print(f"   Total cars processed: {self.total_cars_spawned}")
            print(f"   Cars currently active: {len(self.cars)}")
            print(f"   Cars completed journey: {self.total_cars_exited}")
            print(f"   Congested cells: {final_metrics['congested_cells']}")
            print(f"   Traffic efficiency: {final_metrics['efficiency']:.2f}")
            print(f"   Final congestion level: {final_metrics['congestion_level']:.2f}")
        
        pygame.quit()
        return True
    
    def _save_final_metrics(self):
        """Save comprehensive final analysis"""
        filename = "final_traffic_analysis.txt"
        
        with open(filename, 'w') as f:
            f.write("HYDERABAD TRAFFIC SIMULATION - FINAL ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Configuration
            f.write("SIMULATION CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Original Map Size: {self.original_width} x {self.original_height} pixels\n")
            f.write(f"Simulation Grid: {self.width} x {self.height} cells\n")
            f.write(f"Scale Factor: 1:{self.scale_factor}\n")
            f.write(f"Road Network: {np.sum(self.road_grid)} cells ({100*np.sum(self.road_grid)/(self.width*self.height):.1f}% coverage)\n")
            f.write(f"Traffic Lights: {len(self.traffic_lights)}\n")
            f.write(f"Entry Points: {len(self.entry_points)}\n")
            f.write(f"Max Cars Per Cell: {self.MAX_CARS_PER_CELL}\n")
            f.write(f"Max Total Cars: {self.max_total_cars}\n")
            f.write(f"Congestion Threshold: {self.CONGESTION_THRESHOLD}\n")
            
            # Results summary
            f.write(f"\nTRAFFIC FLOW RESULTS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Simulation Steps: {self.step_count}\n")
            f.write(f"Total Cars Spawned: {self.total_cars_spawned}\n")
            f.write(f"Cars Completed Journey: {self.total_cars_exited}\n")
            f.write(f"Cars Still Active: {len(self.cars)}\n")
            
            if self.total_cars_spawned > 0:
                completion_rate = self.total_cars_exited / self.total_cars_spawned
                f.write(f"Journey Completion Rate: {completion_rate:.1%}\n")
            
            # Performance analysis
            if self.metrics_history:
                cars_data = [m['total_cars'] for m in self.metrics_history]
                congestion_data = [m['congestion_level'] for m in self.metrics_history]
                efficiency_data = [m['efficiency'] for m in self.metrics_history]
                
                f.write(f"\nPERFORMANCE STATISTICS:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Average Active Cars: {np.mean(cars_data):.1f}\n")
                f.write(f"Peak Active Cars: {max(cars_data)}\n")
                f.write(f"Average Congestion Level: {np.mean(congestion_data):.3f}\n")
                f.write(f"Peak Congestion Level: {max(congestion_data):.3f}\n")
                f.write(f"Final Traffic Efficiency: {efficiency_data[-1]:.3f}\n")
                
                # Traffic flow analysis
                f.write(f"\nTRAFFIC FLOW ANALYSIS:\n")
                f.write("-" * 25 + "\n")
                
                high_congestion_steps = sum(1 for c in congestion_data if c > 0.7)
                medium_congestion_steps = sum(1 for c in congestion_data if 0.3 < c <= 0.7)
                low_congestion_steps = sum(1 for c in congestion_data if c <= 0.3)
                
                f.write(f"High Congestion (>70%): {high_congestion_steps} steps ({100*high_congestion_steps/len(congestion_data):.1f}%)\n")
                f.write(f"Medium Congestion (30-70%): {medium_congestion_steps} steps ({100*medium_congestion_steps/len(congestion_data):.1f}%)\n")
                f.write(f"Low Congestion (<30%): {low_congestion_steps} steps ({100*low_congestion_steps/len(congestion_data):.1f}%)\n")
                
                # Recent performance (last 50 steps)
                f.write(f"\nRECENT PERFORMANCE DATA (Last 50 Steps):\n")
                f.write("-" * 40 + "\n")
                f.write("Step,Cars,Moving,Blocked,Congestion,Efficiency\n")
                
                for metrics in self.metrics_history[-50:]:
                    f.write(f"{metrics['step']},{metrics['total_cars']},"
                           f"{metrics['moving_cars']},{metrics['blocked_cars']},"
                           f"{metrics['congestion_level']:.3f},{metrics['efficiency']:.3f}\n")
        
        print(f"📊 Comprehensive analysis saved to {filename}")

def main():
    """Main function to run the multi-car pygame simulation"""
    try:
        print("🚦 HYDERABAD MULTI-CAR TRAFFIC SIMULATION")
        print("Advanced Cellular Automata with Multi-Car Cells & Real-Time Visualization")
        print("=" * 70)
        
        # Initialize the multi-car simulation
        sim = OptimizedTrafficSimulation("hyd_bw.png", "hyd.png")
        
        print(f"\n🎯 New Multi-Car Features:")
        print(f"   ✅ Up to {sim.MAX_CARS_PER_CELL} cars per cell")
        print(f"   ✅ Purple congestion visualization (≥{sim.CONGESTION_THRESHOLD} cars)")
        print(f"   ✅ 300 cars spawn immediately at startup")
        print(f"   ✅ {sim.cars_per_second} new cars added every second")
        print(f"   ✅ Traffic lights concentrated at bottom of map")
        print(f"   ✅ Real-time pygame simulation")
        print(f"   ✅ Interactive controls (pause, reset, exit)")
        print(f"   ✅ Dynamic car density visualization")
        print(f"   ✅ Enhanced spawning system")
        print(f"   ✅ Live performance metrics")
        
        # Run the interactive simulation
        success = sim.run_pygame_simulation()
        
        if success:
            print(f"\n🎉 Multi-car traffic simulation completed successfully!")
            print(f"🗺️  Demonstrated realistic high-density traffic flow")
            print(f"🚗 Multi-car cellular automata with congestion management")
            print(f"📊 Real-time visualization and performance analytics")
        
        return success
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure hyd_bw.png and hyd.png are in the current directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🏁 Multi-car simulation completed successfully!")
    else:
        print(f"\n❌ Simulation failed. Please check the error messages above.")
