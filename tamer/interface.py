import os
import pygame
import cv2
import mediapipe as mp


class Interface_Keyboard:
    """ Pygame interface for training TAMER + RL"""

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        self.screen = pygame.display.set_mode((200, 100))
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

    def get_scalar_feedback(self):
        """
        Get human input. 'W' key for positive, 'A' key for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        area = None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    area = self.screen.fill((0, 255, 0))
                    reward = 1
                    break
                elif event.key == pygame.K_a:
                    area = self.screen.fill((255, 0, 0))
                    reward = -1
                    break
        pygame.display.update(area)
        return reward

    def show_action(self, action):
        """
        Show agent's action on pygame screen
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)


class Interface_Video:
    """ Pygame interface for training TAMER + RL with gesture feedback """

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # Set position of Pygame window and make it larger (e.g., 800x600)
        # Move the window more to the left (e.g., at position "0,100" for the top-left corner of the screen)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,100"  # Position the window closer to the left side of the screen
        os.environ["SDL_VIDEO_CENTERED"] = "0"  # Disable centering of the window

        self.screen_width = 800  # Width of the window
        self.screen_height = 600  # Height of the window
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

        # Initialize MediaPipe Hand detector
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Start webcam
        self.cap = cv2.VideoCapture(0)

        # Set webcam resolution (e.g., 1280x720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_scalar_feedback(self):
        """
        Get human input using the webcam. Detect thumbs up or thumbs down for feedback.
        Returns: scalar reward (1 for thumbs up, -1 for thumbs down, 0 for no feedback)
        """
        reward = 0

        # Capture image from webcam
        ret, frame = self.cap.read()

        if not ret:
            print("Failed to capture image")
            return reward

        # Rotate the frame to fix the orientation (rotate 90 degrees counter-clockwise)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Get the original height and width of the frame
        frame_height, frame_width = frame.shape[:2]

        # Calculate the new height and width to maintain the aspect ratio
        aspect_ratio = frame_width / frame_height
        new_width = self.screen_width  # Keep the width fixed to the window width
        new_height = int(new_width / aspect_ratio)  # Calculate the new height based on the aspect ratio

        # If the new height exceeds the window height, adjust the width accordingly
        if new_height > self.screen_height:
            new_height = self.screen_height
            new_width = int(new_height * aspect_ratio)

        # Resize the frame while maintaining the aspect ratio
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check the position of the thumb to detect thumbs up or thumbs down
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
                
                # Check the rest of the fingers to detect if the hand is closed
                # If any of the fingers are fully closed, we ignore the action
                is_hand_closed = self.is_hand_closed(hand_landmarks)

                # Only detect thumbs up or thumbs down if the hand is not closed
                if not is_hand_closed:
                    if thumb_tip.x < thumb_ip.x:  # Thumb is up
                        reward = 1
                        self.screen.fill((0, 255, 0))  # Green for thumbs up
                    elif thumb_tip.x > thumb_ip.x:  # Thumb is down
                        reward = -1
                        self.screen.fill((255, 0, 0))  # Red for thumbs down

                # Draw hand landmarks on frame for feedback
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Create a surface from the resized frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.surfarray.make_surface(frame)

        # Center the frame in the window
        x_offset = (self.screen_width - new_width) // 2
        y_offset = (self.screen_height - new_height) // 2

        # Display the frame in the Pygame window
        self.screen.blit(frame, (x_offset, y_offset))
        pygame.display.update()

        return reward

    def is_hand_closed(self, hand_landmarks):
        """
        Checks if the hand is closed by comparing the relative positions of the fingers.
        Returns True if the hand is closed, False otherwise.
        """
        # Check if the distance between tips of the fingers is too small, indicating a closed hand
        # We will check the tips of the index and middle fingers as a basic check
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # If the distance between the index and middle fingers is too small, it's likely a closed hand
        distance = ((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2) ** 0.5

        return distance < 0.05  # Threshold can be adjusted if needed

    def show_action(self, action):
        """
        Show agent's action on pygame screen.
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)

    def close(self):
        """ Close webcam and pygame window """
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
    def show_action(self, action):
        """
        Show agent's action on pygame screen.
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)

    def close(self):
        """ Close webcam and pygame window """
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
