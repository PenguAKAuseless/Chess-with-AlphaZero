import torch
import chess
import os
from datetime import datetime

# Input Convolutional Layer
class InputConvolutionalLayer(torch.nn.Module):
    def __init__(self, in_channels=103, out_channels=256):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def load_from_hdf5(self, h5_group):
        self.conv.weight.data = torch.tensor(h5_group['conv_weight'][:])
        self.conv.bias.data = torch.tensor(h5_group['conv_bias'][:])
        self.bn.weight.data = torch.tensor(h5_group['bn_weight'][:])
        self.bn.bias.data = torch.tensor(h5_group['bn_bias'][:])
        self.bn.running_mean.data = torch.tensor(h5_group['bn_running_mean'][:])
        self.bn.running_var.data = torch.tensor(h5_group['bn_running_var'][:])

# Residual Layer
class ResidualLayer(torch.nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=channels)
        self.relu2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = residual + x
        residual = self.relu2(residual)
        return residual
    
    def load_from_hdf5(self, h5_group):
        self.conv1.weight.data = torch.tensor(h5_group['conv1_weight'][:])
        self.conv1.bias.data = torch.tensor(h5_group['conv1_bias'][:])
        self.bn1.weight.data = torch.tensor(h5_group['bn1_weight'][:])
        self.bn1.bias.data = torch.tensor(h5_group['bn1_bias'][:])
        self.bn1.running_mean.data = torch.tensor(h5_group['bn1_running_mean'][:])
        self.bn1.running_var.data = torch.tensor(h5_group['bn1_running_var'][:])
        self.conv2.weight.data = torch.tensor(h5_group['conv2_weight'][:])
        self.conv2.bias.data = torch.tensor(h5_group['conv2_bias'][:])
        self.bn2.weight.data = torch.tensor(h5_group['bn2_weight'][:])
        self.bn2.bias.data = torch.tensor(h5_group['bn2_bias'][:])
        self.bn2.running_mean.data = torch.tensor(h5_group['bn2_running_mean'][:])
        self.bn2.running_var.data = torch.tensor(h5_group['bn2_running_var'][:])

# Value Head
class ValueHead(torch.nn.Module):
    def __init__(self, in_channels, hidden_size):
        super(ValueHead, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.fcl1 = torch.nn.Linear(64, hidden_size)
        self.fcl2 = torch.nn.Linear(hidden_size, 1)
        self.activation = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        x = x.view(-1, 64)
        x = self.activation(self.fcl1(x))
        x = self.tanh(self.fcl2(x))
        return x
    
    def load_from_hdf5(self, h5_group):
        self.conv.weight.data = torch.tensor(h5_group['conv_weight'][:])
        self.conv.bias.data = torch.tensor(h5_group['conv_bias'][:])
        self.bn.weight.data = torch.tensor(h5_group['bn_weight'][:])
        self.bn.bias.data = torch.tensor(h5_group['bn_bias'][:])
        self.bn.running_mean.data = torch.tensor(h5_group['bn_running_mean'][:])
        self.bn.running_var.data = torch.tensor(h5_group['bn_running_var'][:])
        self.fcl1.weight.data = torch.tensor(h5_group['fcl1_weight'][:])
        self.fcl1.bias.data = torch.tensor(h5_group['fcl1_bias'][:])
        self.fcl2.weight.data = torch.tensor(h5_group['fcl2_weight'][:])
        self.fcl2.bias.data = torch.tensor(h5_group['fcl2_bias'][:])

# Policy Head
class PolicyHead(torch.nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, 2, kernel_size=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fcl = torch.nn.Linear(in_features=8 * 8 * 2, out_features=8 * 8 * (56 + 8 + 9))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fcl(x)
        return x
    
    def load_from_hdf5(self, h5_group):
        self.conv.weight.data = torch.tensor(h5_group['conv_weight'][:])
        self.conv.bias.data = torch.tensor(h5_group['conv_bias'][:])
        self.bn.weight.data = torch.tensor(h5_group['bn_weight'][:])
        self.bn.bias.data = torch.tensor(h5_group['bn_bias'][:])
        self.bn.running_mean.data = torch.tensor(h5_group['bn_running_mean'][:])
        self.bn.running_var.data = torch.tensor(h5_group['bn_running_var'][:])
        self.fcl.weight.data = torch.tensor(h5_group['fcl_weight'][:])
        self.fcl.bias.data = torch.tensor(h5_group['fcl_bias'][:])

# Deep Neural Network
class DeepNeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels=103, out_channels=256, num_residual=10, hidden_size=256):
        super().__init__()
        self.conv = InputConvolutionalLayer(in_channels=in_channels, out_channels=out_channels)
        self.residual_list = torch.nn.Sequential(*[ResidualLayer(channels=out_channels) for _ in range(num_residual)])
        self.value_head = ValueHead(in_channels=out_channels, hidden_size=hidden_size)
        self.policy_head = PolicyHead(channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.residual_list(x)
        value_x = x.clone()
        policy_x = x
        return self.value_head(value_x), self.policy_head(policy_x)
    
    def load_from_hdf5(self, filename):
        import h5py
        with h5py.File(filename, 'r') as f:
            self.conv.load_from_hdf5(f['conv'])
            for i, residual in enumerate(self.residual_list):
                residual.load_from_hdf5(f[f'residual_{i}'])
            self.value_head.load_from_hdf5(f['value_head'])
            self.policy_head.load_from_hdf5(f['policy_head'])

# Chess State
class ChessState:
    def __init__(self, board: chess.Board):
        self.board = board

    def clone(self):
        return ChessState(self.board.copy())

    def apply_move(self, move: chess.Move):
        self.board.push(move)

    def is_terminal(self):
        return self.board.is_game_over()

    def legal_moves(self):
        return list(self.board.legal_moves)

    def result(self):
        return self.board.result()

# Board to Tensor Conversion
def boards_to_tensor(history):
    import numpy as np
    tensor = np.zeros((103, 8, 8), dtype=np.float32)
    piece_planes = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,
    }
    for t, chess_state in enumerate(history[-8:]):
        board = chess_state.board
        offset = t * 12
        for square, piece in board.piece_map().items():
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            plane = offset + piece_planes[piece.symbol()]
            tensor[plane, row, col] = 1
    latest_board = history[-1].board
    tensor[96, :, :] = int(latest_board.turn)
    tensor[97, :, :] = int(latest_board.has_kingside_castling_rights(chess.WHITE))
    tensor[98, :, :] = int(latest_board.has_queenside_castling_rights(chess.WHITE))
    tensor[99, :, :] = int(latest_board.has_kingside_castling_rights(chess.BLACK))
    tensor[100, :, :] = int(latest_board.has_queenside_castling_rights(chess.BLACK))
    tensor[101, :, :] = latest_board.halfmove_clock / 100.0
    tensor[102, :, :] = latest_board.fullmove_number / 100.0
    return tensor

# Monte Carlo Tree Node
class MonteCarloTreeNode:
    def __init__(self, state: ChessState, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def expand(self, move_priors):
        for move, prior in move_priors.items():
            next_state = self.state.clone()
            next_state.apply_move(move)
            self.children[move] = MonteCarloTreeNode(next_state, parent=self, prior=prior)

# Evaluator
class Evaluator:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def evaluate(self, history):
        import torch.nn.functional as F
        tensor = boards_to_tensor(history)
        input_tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value, policy_logits = self.model(input_tensor)

        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.item()

        legal_moves = history[-1].legal_moves()
        move_priors = {}
        for move in legal_moves:
            idx = self.move_to_index(move)
            move_priors[move] = policy[idx]

        return move_priors, value

    def move_to_index(self, move: chess.Move) -> int:
        from_square = move.from_square
        to_square = move.to_square
        from_row = 7 - chess.square_rank(from_square)
        from_col = chess.square_file(from_square)
        to_row = 7 - chess.square_rank(to_square)
        to_col = chess.square_file(to_square)

        delta_row = to_row - from_row
        delta_col = to_col - from_col

        if move.promotion is None:
            directions = [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            for dir_idx, (dr, dc) in enumerate(directions):
                for step in range(1, 8):
                    if delta_row == dr * step and delta_col == dc * step:
                        plane = dir_idx * 7 + (step - 1)
                        return (from_row * 8 + from_col) * 73 + plane

        knight_deltas = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for i, (dr, dc) in enumerate(knight_deltas):
            if delta_row == dr and delta_col == dc:
                return (from_row * 8 + from_col) * 73 + 56 + i

        if move.promotion in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}:
            if (from_row == 1 and to_row == 0) or (from_row == 6 and to_row == 7):
                direction_map = {0: 0, -1: 1, 1: 2}
                dir = to_col - from_col
                if dir in direction_map:
                    dir_idx = direction_map[dir]
                    piece_map = {
                        chess.KNIGHT: 0,
                        chess.BISHOP: 1,
                        chess.ROOK: 2,
                        chess.QUEEN: 3
                    }
                    piece_idx = piece_map[move.promotion]
                    plane = 64 + dir_idx * 4 + piece_idx
                    return (from_row * 8 + from_col) * 73 + plane
                else:
                    raise ValueError(f"Invalid promotion direction for move: {move}, delta_col: {dir}")
            else:
                raise ValueError(f"Invalid promotion position for move: {move}, from: {from_row},{from_col} to: {to_row},{to_col}")
        raise ValueError(f"Invalid or unsupported move for AlphaZero mapping: {move}")

# Monte Carlo Search Tree
class MonteCarloSearchTree:
    def __init__(self, evaluator, simulations=800, c_puct=1.0):
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, state: ChessState, history):
        import numpy as np
        root = MonteCarloTreeNode(state=state)
        move_priors, value = self.evaluator.evaluate(history)
        root.expand(move_priors)
        for _ in range(self.simulations):
            node = root
            path = [node]
            current_history = history.copy()
            while node.is_expanded():
                move, node = self.select(node)
                path.append(node)
                current_state = node.state.clone()
                current_history.append(current_state)
                current_history = current_history[-8:]
            if not node.state.is_terminal():
                move_priors, value = self.evaluator.evaluate(current_history)
                node.expand(move_priors)
            self.backpropagate(path, value)
        return self.select_action(root)

    def select(self, node):
        import numpy as np
        best_score = -float("inf")
        best_move = None
        best_child = None
        for move, child in node.children.items():
            ucb = self.ucb_score(node, child)
            if ucb > best_score:
                best_score = ucb
                best_move = move
                best_child = child
        return best_move, best_child

    def ucb_score(self, parent, child):
        import numpy as np
        q = child.value()
        u = self.c_puct * child.prior * (np.sqrt(parent.visits) / (1 + child.visits))
        return q + u

    def backpropagate(self, path, value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def select_action(self, root):
        move_visits = [(move, child.visits) for move, child in root.children.items()]
        move_visits.sort(key=lambda x: x[1], reverse=True)
        return move_visits[0][0]

# Interactive Game Functions
def get_valid_user_move(board):
    while True:
        try:
            user_input = input("Enter your move (e.g., e2e4) or 'resign' to concede: ").strip().lower()
            if user_input == 'resign':
                return None
            move = chess.Move.from_uci(user_input)
            if move in board.legal_moves:
                return move
            else:
                print("Invalid move. Please enter a legal move (e.g., e2e4).")
        except ValueError:
            print("Invalid input format. Use UCI notation (e.g., e2e4).")

def print_board(board):
    piece_map = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    print("\n   +-----------------+")
    for rank in range(7, -1, -1):
        print(f" {rank + 1} |", end=" ")
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            symbol = piece_map.get(piece.symbol(), '.') if piece else '.'
            print(symbol, end=" ")
        print("|")
    print("   +-----------------+")
    print("     a b c d e f g h\n")

def log_game_result(log_dir, opponent, ai_side, result, moves):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"game_{timestamp}.txt")
    with open(filename, 'w') as f:
        f.write(f"Opponent: {opponent}\n")
        f.write(f"AI Side: {'White' if ai_side == chess.WHITE else 'Black'}\n")
        f.write(f"Result: {result}\n")
        f.write("Moves:\n")
        for i, move in enumerate(moves, 1):
            f.write(f"{i}. {move.uci()}\n")
    print(f"Game result logged to {filename}")

def play_against_user(model, device, ai_side, log_dir, checkpoint_dir="checkpoint"):
    # Initialize model and evaluator
    model.eval()
    evaluator = Evaluator(model, device)
    mcts = MonteCarloSearchTree(evaluator, simulations=800, c_puct=1.0)
    
    # Initialize game
    board = chess.Board()
    state = ChessState(board)
    history = [state.clone()]
    moves = []
    
    print(f"AI plays as {'White' if ai_side == chess.WHITE else 'Black'}.")
    print("Enter moves in UCI notation (e.g., e2e4). Type 'resign' to concede.")
    
    while not state.is_terminal():
        print_board(state.board)
        if state.board.turn == ai_side:
            move = mcts.run(state, history)
            print(f"AI move: {move.uci()}")
        else:
            move = get_valid_user_move(state.board)
            if move is None:
                print("You resigned.")
                log_game_result(log_dir, "human", ai_side, "AI wins by resignation", moves)
                return 'AI wins by resignation'
        moves.append(move)
        state.apply_move(move)
        history.append(state.clone())
        history = history[-8:]
    
    print_board(state.board)
    result = state.result()
    if result == '1-0':
        winner = 'White' if ai_side == chess.BLACK else 'AI'
    elif result == '0-1':
        winner = 'Black' if ai_side == chess.WHITE else 'AI'
    else:
        winner = 'Draw'
    print(f"Game over. Result: {result} ({winner})")
    log_game_result(log_dir, "human", ai_side, f"{result} ({winner})", moves)
    return result

if __name__ == "__main__":
    import random
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepNeuralNetwork(in_channels=103, out_channels=256, num_residual=10, hidden_size=256).to(device)
    checkpoint_dir = "checkpoint"
    log_dir = "logs"

    # Load latest checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_from_hdf5(checkpoint_path)
    else:
        print("No checkpoints found. Please ensure a trained model checkpoint exists in the 'checkpoint' directory.")
        exit(1)

    # User chooses AI side
    side = input("Choose AI side (white/black/random): ").strip().lower()
    while side not in ['white', 'black', 'random']:
        print("Invalid choice. Please choose 'white', 'black', or 'random'.")
        side = input("Choose AI side (white/black/random): ").strip().lower()

    ai_side = chess.WHITE if side == 'white' else chess.BLACK if side == 'black' else random.choice([chess.WHITE, chess.BLACK])

    # Play the game
    result = play_against_user(model, device, ai_side, log_dir)