from neuralnets import dawn
import numpy as np
import itertools


# Current Best BYTE: [9,56,56,18,9],      sigmoid,   0.012 *= 1      15000: 0.002533
# Current Best DAWN: [9,128,128,64,32,9], relu-relu, 0.22 *= 0.99995 15000: 0.000647


dawn = dawn([9,128,128,64,32,9], "relu", "relu")
dawn.init_weights("xavier_normal")
epochs = 50000
learn_rate = 0.22
decay = 0.99995
checks = 100
board = [0,0,0,0,0,0,0,0,0]
current_player = -1
x = []
y = []

def check_win(board):
  lines = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
  ]
  for line in lines:
    total = sum(board[i] for i in line)
    if total  == 3:
      return 1
    elif total == -3:
      return -1
  if 0 not in board:
    return 0
  return None

def train_dawn():
  x.clear()
  y.clear()
  seen = {}
  def generate(board, player):
    winner = check_win(board)
    if winner is not None:
      return [winner * player]
    board_tuple = tuple(board)
    x_count = board.count(1)
    o_count = board.count(-1)
    if abs(x_count - o_count) > 1:
      return []
    if board_tuple in seen:
      return seen[board_tuple]
    outcomes = []
    for i in range(9):
      if board[i] != 0:
        continue
      board[i] = player
      result = generate(board, -player)
      if isinstance(result, int):
        result = [result]
      if not result:
        result = [0]
      outcomes.append(list(result))
      board[i] = 0
    target = [0,0,0,0,0,0,0,0,0]
    move = 0
    for i in range(9):
      if board[i] != 0:
        continue
      outcome = outcomes[move]
      if any(o == player for o in outcome):
        target[i] = 1.0
      elif any(o == -player for o in outcome):
        target[i] = 1.0
      else:
        target[i] = 0.0
      move += 1
    x.append(board.copy())
    y.append(target)
    seen[board_tuple] = target
    return target
  generate([0,0,0,0,0,0,0,0,0], -1)
  x_arr = np.array(x)
  y_arr = np.array(y)
  x_arr = x_arr.T
  y_arr = y_arr.T
  dawn.teach(x_arr, y_arr, learn_rate, "backprop", epochs, checks, decay)

def print_board(board):
  symbols = {1:"[X]", -1:"[O]"}
  for i in range(3):
    row = []
    for j in range(3):
      k = j + i*3
      if board[k] == 0:
        row.append(" " + str(k) + " ")
      else:
        row.append(symbols[board[k]])
    print("|".join(row))
    if i < 2:
      print("---+---+---")

def human_move(board):
  move = -1
  while move not in range(9) or board[move] != 0:
    move = int(input(f"[~/dawn_t3.py] Your Turn! (0-8): "))
  return move

def dawn_move(board):
  output = dawn.think(np.array(board).reshape(9,1)).flatten()
  masked = [o if board[i] == 0 else -1e9 for i, o in enumerate(output)]
  move = masked.index(max(masked))
  return move

train_dawn()

while True:
  if current_player == 1:
    print("")
    print_board(board)
    print("")
    move = human_move(board)
  else:
    move = dawn_move(board)
  board[move] = current_player if current_player == 1 else -1
  winner = check_win(board)
  if winner is not None:
    print("")
    print_board(board)
    print("")
    if winner == 1:
      print("[~/dawn_t3.py] You Win! (^C To Exit Game)")
    elif winner == -1:
      print("[~/dawn_t3.py] Dawn Wins! (^C To Exit Game)")
    else:
      print("[~/dawn_t3.py] Draw! (^C To Exit Game)")
    board = [0,0,0,0,0,0,0,0,0]
    current_player = -1
  else:
    current_player *= -1
