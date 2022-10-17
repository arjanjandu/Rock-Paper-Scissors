import random
import cv2
from keras.models import load_model
import numpy as np

class RPS:
    def __init__(self, choices = [], player = [''], computer = [''], computer_lives=int, player_lives=int, winner=int, num_lives=int ):
        self.choices = choices
        self.player = player
        self.computer = computer
        self.computer_lives = computer_lives
        self.player_lives = player_lives
        self.winner = winner
        self.num_lives = num_lives
        
    
    def get_comp_choice(self):
        self.computer = random.choice(self.choices)
      
        
    
    def get_user_prediction(self):
        model = load_model('keras_model.h5')
        cap = cv2.VideoCapture(0)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        while True: 
            ret, frame = cap.read()
            resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
            data[0] = normalized_image
            self.prediction = model.predict(data)
            cv2.imshow('frame', frame)
            # Press q to close the window
            np.argmax(self.prediction) 
            if cv2.waitKey(5000) & 0xFF == ord('q'): 
                self.get_user_choice()
            else:
                break
                            
        # After the loop release the cap object
        cap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
    
    def get_user_choice(self):   
        self.get_user_prediction()
        if np.argmax(self.prediction) == 0:
            self.player = 'rock'
        elif np.argmax(self.prediction) == 1:
            self.player = 'paper'
        elif np.argmax(self.prediction) == 2:
            self.player = 'scissors'
        else:
            self.player = 'nothing'
        

    def get_winner(self):
        if self.player == 'rock' and  self.computer == 'rock':
           print('Draw\n')
           self.winner = 0 
        elif self.player == 'paper'  and  self.computer == 'paper':
           print('Draw\n')
           self.winner = 0 
        elif self.player == 'scissor' and  self.computer == 'scissors':
           print('Draw\n')
           self.winner= 0
        elif self.player == 'rock' and self.computer == 'paper':
           print('computer wins\n')
           self.winner = 1
        elif self.player == 'rock' and self.computer == 'scissors':
           print('You win\n')
           self.winner = 2
        elif self.player == 'paper' and self.computer == 'rock':
           print('You win\n')
           self.winner = 2
        elif self.player == 'paper' and self.computer == 'scissors':
           print('computer wins\n')
           self.winner = 1
        elif self.player == 'scissors' and self.computer == 'paper':
           print('You win\n')
           self.winner = 2
        elif self.player == 'scissors' and self.computer == 'rock':
           print('computer wins\n')
           self.winner = 1
        else:
            print('incorrect input, try again\n')
        

    def final(self):
        self.get_comp_choice()
        self.get_user_choice()
        self.get_winner()
        print(f'computer chose: {self.computer}') 
        if self.winner == 1:
         self.computer_lives = self.computer_lives + 1 
         self.play()
        elif self.winner == 2:
         self.player_lives = self.player_lives + 1
         self.play()
        else: 
         self.play()
    
        
        
    def play(self): 
        print(f'\nRock, Paper or Scissors: \n\nComputer live : {self.computer_lives} \nPlayer lives: {self.player_lives} \n')
        if self.player_lives == self.num_lives:
         print('\nYou win the game')
        elif self.computer_lives == self.num_lives:
         print('\nComputer wins')
        else: 
         self.final()
        

if __name__ == '__main__':
    choices = ['rock', 'paper', 'scissors']
    game = RPS(choices, computer_lives=0, player_lives=0, winner=0, num_lives=3)
    game.play()


    