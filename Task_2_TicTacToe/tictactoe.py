import math

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.human = 'X'
        self.ai = 'O'
    
    def display_board(self):
        print('\n')
        for i in range(3):
            print(f' {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]} ')
            if i < 2:
                print('-----------')
        print('\n')
    
    def is_winner(self, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in win_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False
    
    def is_board_full(self):
        return ' ' not in self.board
    
    def get_available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def minimax(self, depth, is_maximizing):
        if self.is_winner(self.ai):
            return 10 - depth
        elif self.is_winner(self.human):
            return depth - 10
        elif self.is_board_full():
            return 0
        
        if is_maximizing:
            max_score = -math.inf
            for move in self.get_available_moves():
                self.board[move] = self.ai
                score = self.minimax(depth + 1, False)
                self.board[move] = ' '
                max_score = max(score, max_score)
            return max_score
        else:
            min_score = math.inf
            for move in self.get_available_moves():
                self.board[move] = self.human
                score = self.minimax(depth + 1, True)
                self.board[move] = ' '
                min_score = min(score, min_score)
            return min_score
    
    def ai_move(self):
        best_score = -math.inf
        best_move = None
        
        for move in self.get_available_moves():
            self.board[move] = self.ai
            score = self.minimax(0, False)
            self.board[move] = ' '
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def play(self):
        print("Tic-Tac-Toe: You are X, AI is O")
        print("Positions are numbered 0-8 (left to right, top to bottom)")
        
        while True:
            self.display_board()
            
            if self.is_winner(self.ai):
                print("AI wins!")
                break
            elif self.is_winner(self.human):
                print("You win!")
                break
            elif self.is_board_full():
                print("It's a draw!")
                break
            
            move = int(input("Enter your move (0-8): "))
            if move not in self.get_available_moves():
                print("Invalid move!")
                continue
            
            self.board[move] = self.human
            
            if self.is_winner(self.human):
                self.display_board()
                print("You win!")
                break
            
            ai_move = self.ai_move()
            self.board[ai_move] = self.ai
            print(f"AI plays at position {ai_move}")

if __name__ == "__main__":
    game = TicTacToe()
    game.play()
