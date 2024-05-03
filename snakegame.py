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
import sys
import random
import time

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

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

        #Goes from 0,0 to 14, 14 maps onto x y when mutiplying with the board (by 50 since image is 700x700), is a 2d array
        self.location
        #Just int, determines how far back path goes
        self.length
        #1d array
        self.path



class Food:
    """
    Food class to tally score
    """
    def __init__(self):
        #randon location on borad, but not where 
        self.location

    def respawn():
        Xlocation = random
        Ylocation = random
        location[x][y] = random






class Game:
    """
    Main game loop. Runs until the 
    user presses "q".
    """

    BACKGROUND_COLOR = (0, 0, 0)


    def __init__(self):
        # Load game elements
        # self.score

        # self.food

        # self.snake


        self.HEIGHT = 700
        self.WIDTH = 700

        # Create the game window
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Snake Game")

        self.image = pygame.image.load("data/Snake_Game_Grid.png")
        self.screen.blit(self.image, (0,0))

        self.board




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
                pixelCoordThumb = DrawingUtil._normalized_to_pixel_coordinates(thumb.x, thumb.y, imageWidth, imageHeight)

                if pixelCoord:
                    cv2.circle(image, (pixelCoord[0], pixelCoord[1]), 25, GREEN, 5)
                    #self.check_enemy_intercept_hitsametime(pixelCoord[0], pixelCoord[1], self.green_enemy, image, time.time())
                #     # for enemy in self.enemies:
                #     #     self.check_enemy_intercept(pixelCoord[0], pixelCoord[1], enemy, image)
                #     self.check_enemy_intercept(pixelCoord[0], pixelCoord[1], self.green_enemy, image)
            #####DRAW FINGER
            
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    

        # Draw the initial screen
        #self.screen.fill(self.BACKGROUND_COLOR)
        #self.screen.draw(self.screen)
        #self.mango.draw(self.screen)
        self.screen.blit(self.image, (0,0))
        pygame.display.flip()

        # TODO: Modify loop condition  
        running  = True
        while self.video.isOpened() and running:

            self.screen.blit(self.image, (0,0))
            pygame.display.flip()

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The image comes in mirrored - flip it
            image = cv2.flip(image, 1)



            # Draw score onto screen
            # cv2.putText(image, str(self.score), (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

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
            self.draw_landmarks_on_hand(image, results)
            #self.check_enemy_kill(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #FOR VIDEO TO SHOW - AAAAAAAA
            cv2.imshow('Hand Tracking', image)


            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                #print(self.score)
                break

        self.video.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()




if __name__ == "__main__":        
    g = Game()
    g.run()