"""
Simple Interactive Mountain Car Game using Pygame
Allows manual control of the Mountain Car using keyboard inputs
"""

import gymnasium as gym
import pygame
import numpy as np
import sys
import os

# Disable SDL audio to prevent ALSA errors
os.environ['SDL_AUDIODRIVER'] = 'dummy'

def main():
    """Run the interactive Mountain Car game"""
    # Initialize pygame
    pygame.init()
    
    # Game parameters
    step_duration = 0.03  # Time in seconds per step (0.05 = 20 FPS)
    
    # Create environment
    env = gym.make('MountainCar-v0', render_mode='human')
    
    
    # Game loop
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            # Reset environment for new episode
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_done = False
            
            
            while not episode_done and running:
                # Handle events
                action = 1  # Default: do nothing
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                
                # Get current key states for continuous control
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    action = 0  # Push left
                elif keys[pygame.K_RIGHT]:
                    action = 2  # Push right
                else:
                    action = 1  # Do nothing
                
                # Take action in environment
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Render the environment
                env.render()
                
                # Check if episode is done
                if terminated or truncated:
                    episode_done = True
                    success = terminated
                    score = episode_steps  # Lower is better
                    
                    # Show score screen and wait for spacebar
                    show_score_screen(env, success, score, episode_reward)
                    wait_for_spacebar()
                
                # Control game speed using step_duration
                pygame.time.wait(int(step_duration * 1000))  # Convert to milliseconds
                
    except KeyboardInterrupt:
        print("\n👋 Game interrupted by user")
    
    finally:
        # Clean up
        env.close()
        pygame.quit()
        print("🎮 Thanks for playing Mountain Car!")

def show_score_screen(env, success, score, total_reward):
    """Display score screen after episode completion"""
    # Create a simple score display using pygame
    pygame.init()
    
    # Get the screen dimensions from the environment window
    screen = pygame.display.get_surface()
    if screen is None:
        # Fallback if no display surface exists
        screen = pygame.display.set_mode((600, 400))
    
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    
    # Clear screen with semi-transparent overlay
    overlay = pygame.Surface(screen.get_size())
    overlay.set_alpha(200)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))
    
    # Calculate text positions
    width, height = screen.get_size()
    center_x = width // 2
    
    y_pos = height // 4
    
    # Title
    if success:
        title = font_large.render("SUCCESS!", True, GREEN)
        subtitle = font_medium.render("You reached the goal!", True, WHITE)
    else:
        title = font_large.render("TIME UP", True, YELLOW)
        subtitle = font_medium.render("Try again!", True, WHITE)
    
    # Center the title
    title_rect = title.get_rect(center=(center_x, y_pos))
    screen.blit(title, title_rect)
    
    subtitle_rect = subtitle.get_rect(center=(center_x, y_pos + 60))
    screen.blit(subtitle, subtitle_rect)
    
    # Score (steps taken - lower is better)
    score_color = GREEN if success and score < 150 else YELLOW if success else RED
    score_text = font_medium.render(f"Score: {score} steps", True, score_color)
    score_rect = score_text.get_rect(center=(center_x, y_pos + 120))
    screen.blit(score_text, score_rect)
    
    # Total reward
    reward_text = font_small.render(f"Total Reward: {total_reward:.1f}", True, WHITE)
    reward_rect = reward_text.get_rect(center=(center_x, y_pos + 160))
    screen.blit(reward_text, reward_rect)
    
    
    # Instructions
    instruction = font_small.render("Press SPACE to play again, ESC to quit", True, WHITE)
    instruction_rect = instruction.get_rect(center=(center_x, height - 60))
    screen.blit(instruction, instruction_rect)
    
    pygame.display.flip()
    

def wait_for_spacebar():
    """Wait for spacebar press to continue or ESC to quit"""
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        pygame.time.wait(50)  # Small delay to prevent high CPU usage

if __name__ == "__main__":
    main()
