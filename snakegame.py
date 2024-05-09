
"""
Snake Game!

@author: Kayan Sunderam
@version: April 2024

"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import pygame
import random
# Set up the screen
GRID_SIZE = 50
GRID_WIDTH = 14
GRID_HEIGHT = 14
SCREEN_WIDTH = GRID_WIDTH * GRID_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * GRID_SIZE
# SCREEN_WIDTH = 700
# SCREEN_HEIGHT = 700
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")
image = pygame.image.load("data/Snake_Game_Grid.png")
screen.blit(image, (0,0))

# Colors
WHITE = (255, 255, 255)
BLUE = (2, 51, 166) #BLUEEEEEE
RED = (255, 0, 0)

# Initialize Pygame
pygame.init()

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Snake:
    """
    Main character of game
    """
    def __init__(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = BLUE

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + (x * GRID_SIZE)) % SCREEN_WIDTH), (cur[1] + (y * GRID_SIZE)) % SCREEN_HEIGHT)
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset()
        elif self.get_head_position()[0] < 0 or self.get_head_position()[0] >= SCREEN_WIDTH \
                or self.get_head_position()[1] < 0 or self.get_head_position()[1] >= SCREEN_HEIGHT:
            self.reset()
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, WHITE, r, 1)

    def handle_keys(self, xPos, yPos):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.turn(UP)
                elif event.key == pygame.K_DOWN:
                    self.turn(DOWN)
                elif event.key == pygame.K_LEFT:
                    self.turn(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.turn(RIGHT)

        if xPos > 500 and xPos < 2000 and yPos < 400:
            self.turn(UP)
            print("UP")
            print(xPos)
            print(yPos)
        elif xPos > 500 and xPos < 2000 and yPos > 1200:
            self.turn(DOWN)
            print("DOWN")
            print(xPos)
            print(yPos)
        elif yPos > 400 and yPos < 1200 and xPos < 500:
            self.turn(LEFT)
            print("LEFT")
            print(xPos)
            print(yPos)
        elif yPos > 400 and yPos < 1200 and xPos > 2000:
            self.turn(RIGHT)
            print("RIGHT")
            print(xPos)
            print(yPos)



class Food:
    """
    Food class to tally score
    """
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH - 1) * GRID_SIZE, random.randint(0, GRID_HEIGHT - 1) * GRID_SIZE)

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, WHITE, r, 1)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class Game:
    """
    Main game loop. Runs until the 
    user presses "q".
    """


    def __init__(self):
        # Load game elements
        # self.score

        # self.food

        # self.snake


        self.clock = pygame.time.Clock()
        # screen.blit(self.image, (0,0))


        self.HEIGHT = 700
        self.WIDTH = 700

        self.xPos = 0
        self.yPos = 0

        # Create the game window
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Snake Game")

        # self.image = pygame.image.load("data/Snake_Game_Grid.png")
        # self.screen.blit(self.image, (0,0))

        #self.board




        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        # TODO: Load video
        self.video = cv2.VideoCapture(1)

    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())
            
            #DRAW ON FINGER
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
                thumb = hand_landmarks[HandLandmarkPoints.THUMB_TIP.value]

                imageHeight, imageWidth = image.shape[:2]

                pixelCoord = DrawingUtil._normalized_to_pixel_coordinates(finger.x, finger.y, imageWidth, imageHeight)
                # self.xPos = pixelCoord[0]
                # self.yPos = pixelCoord[1]
                pixelCoordThumb = DrawingUtil._normalized_to_pixel_coordinates(thumb.x, thumb.y, imageWidth, imageHeight)

                if pixelCoord is not None:
                    cv2.circle(image, (pixelCoord[0], pixelCoord[1]), 25, BLUE, 5)
                    self.xPos = pixelCoord[0]
                    self.yPos = pixelCoord[1]
                    # print(self.xPos)
                    # print(self.yPos)
                    #self.check_enemy_intercept_hitsametime(pixelCoord[0], pixelCoord[1], self.green_enemy, image, time.time())
                #     # for enemy in self.enemies:
                #     #     self.check_enemy_intercept(pixelCoord[0], pixelCoord[1], enemy, image)
                #     self.check_enemy_intercept(pixelCoord[0], pixelCoord[1], self.green_enemy, image)
            #####DRAW FINGER


def main():
    # Set up the game objects
    snake = Snake()
    food = Food()
    game = Game()

    clock = pygame.time.Clock()
    game.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Game")

    image = pygame.image.load("data/Snake_Game_Grid.png")
    game.screen.blit(image, (0,0))


    # Main game loop
    while game.video.isOpened() and True:
        # screen.fill((0, 0, 0))
        # screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        # pygame.display.set_caption("Snake Game")

        image = pygame.image.load("data/Snake_Game_Grid.png")
        game.screen.blit(image, (0,0))

        snake.handle_keys(game.xPos, game.yPos)
        snake.move()

        if snake.get_head_position() == food.position:
            snake.length += 1
            food.randomize_position()

        snake.draw(game.screen)
        food.draw(game.screen)

        # Check if snake hits the edge of the grid
        if snake.get_head_position()[0] < 0 or snake.get_head_position()[0] >= SCREEN_WIDTH \
                or snake.get_head_position()[1] < 0 or snake.get_head_position()[1] >= SCREEN_HEIGHT:
            snake.reset()

        pygame.display.update()
        clock.tick(10)

        # Get the current frame
        frame = game.video.read()[1]

        # Convert it to an RGB image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # The image comes in mirrored - flip it
        image = cv2.flip(image, 1)



        # Draw score onto screen
        # cv2.putText(image, str(self.score), (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)

        # Convert the image to a readable format and find the hands
        to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = game.detector.detect(to_detect)

        #Draw the enemy on the image
        #self.green_enemy.draw(image)

        #self.red_enemy.draw(image)

        # if time.time() - self.starttime >= 2:
        #     self.enemies.append(Enemy(GREEN))
        #     self.enemies.append(Enemy(RED))
            
        #     self.starttime = time.time()

        # for enemy in self.enemies:
        #     enemy.draw(image)

        # Draw the hand landmarks
        game.draw_landmarks_on_hand(image, results)
        #self.check_enemy_kill(image, results)

        # Change the color of the frame back
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Hand Tracking', 2500, 1600)

        #FOR VIDEO TO SHOW - AAAAAAAA
        #cv2.imshow('Hand Tracking', image)
        cv2.destroyAllWindows
        # print(cv2.getWindowImageRect('Hand Tracking'))

if __name__ == "__main__":
    main()


