from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "Rock",
    1: "Paper",
    2: "Scissor"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def winner_winner(m1,m2):
    if m1 == m2:
        return "Tie"
    
    if m1 == "Rock":
        if m2 == "Paper":
            return "Computer"
        if m2 == "Scissor":
            return "User"
    
    if m1 == "Paper":
        if m2 == "Rock":
            return "User"
        if m2 == "Scissor":
            return "Computer"
    
    if m1 == "Scissor":
        if m2 == "Rock":
            return "Computer"
        if m2 == "Paper":
            return "User"
    

model = load_model("C://Users//a//Desktop//VScode//Rock Paper Scissor//My Game//rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    #Player
    cv2.rectangle(frame, (50, 150), (225, 400), (255, 255, 255), 2)
    #Computer
    cv2.rectangle(frame, (425, 150), (600, 400), (255, 255, 255), 2)

    #Taking Users Action
    roi = frame[150:400, 50:225]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    #Predicting with users Action
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    #Deciding Winner
    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['Rock', 'Paper', 'Scissor'])
            winner = winner_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    #Writing on Screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (10, 50), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (355, 50), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (230, 450), font, 1.0, (0, 0, 255), 4, cv2.LINE_AA)


    if computer_move_name != "none":
        icon = cv2.imread(
            "C://Users//a//Desktop//VScode//Rock Paper Scissor//My Game//Test Images//{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (175, 250))
        frame[150:400, 425:600] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


