# Connect4 Game
import os
import time
# Sets the dimensions of the game board
rows, col = 6, 7

# Initializes the game board with empty spaces
game_board = [[' ' for _ in range(col)] for _ in range(rows)]

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')  # works on Windows/Mac/Linux

def animate_drop(col, piece):
    # Find the lowest empty row
    row = rows - 1
    while row >= 0 and game_board[row][col] != ' ':
        row -= 1
    if row < 0:
        return  # column is full
    
    # Animate falling piece
    for r in range(row + 1):  
        clear()
        temp_board = [row.copy() for row in game_board]  # copy current board
        temp_board[r][col] = piece
        for row_ in temp_board:
            print('| ' + " | ".join(row_) + " |")
        print("  " + "   ".join(str(i) for i in range(1, len(game_board[0]) + 1)))
        time.sleep(0.1)

    # Finalize placement in actual game board
    game_board[row][col] = piece
    
# Function to print the current state of the game board
def print_board():
    for row in game_board:
        print('| ' + " | ".join(row) + " |")
    print("  " + "   ".join(str(i) for i in range(1, len(game_board[0]) + 1)))

def x_user_turn():
    turn = int(input("Player 1 turn, where would you like to go?  ")) - 1
    animate_drop(turn, 'X')

def o_user_turn():
    turn = int(input("Player 2 turn, where would you like to go?  ")) - 1
    animate_drop(turn, 'O')

# Function to check for horizontal wins
def check_horizontal_win():
    for row in game_board:
        check = ''
        for item in row:
            check += str(item)
        if 'XXXX' in check:
            print("X is Winner!")
            exit()
        elif 'OOOO' in check:
            print("O is Winner!")
            exit()

# Function to check for vertical wins
def check_vertical_win():
    for j in range(7):
        check = ''
        for i in range(6):
            check += str(game_board[i][j])
        if 'XXXX' in check:
            print("X is Winner!")
            exit()
        elif 'OOOO' in check:
            print("O is Winner!")
            exit()

# Function to check for diagonal wins (both directions)
def check_diag_win():
    for j in range(3):
        for i in range(4):
            left_diagonal = ''
            right_diagonal = ''
            for k in range(4):
                left_diagonal += str(game_board[j+k][i+k])
                right_diagonal += str(game_board[j+k][3+i-k])
            if left_diagonal == 'XXXX' or right_diagonal == 'XXXX':
                print("X is Winner!")
                exit()
            elif left_diagonal == 'OOOO' or right_diagonal == 'OOOO':
                print("O is Winner!")
                exit()    

# Function to check for any type of win         
def check_win():
    check_horizontal_win()
    check_vertical_win()
    check_diag_win()

# Main gameplay loop
def gameplay():
    print_board()
    while True:
        check_win()
        x_user_turn()
        check_win()
        o_user_turn()

# Start the game
gameplay()
    