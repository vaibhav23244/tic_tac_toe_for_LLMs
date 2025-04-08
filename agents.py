from typing import Tuple
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq

def get_model_provider(provider: str, model_name: str):
    """
    Creates and returns the appropriate model instance based on the provider.

    Args:
        provider: The model provider ('groq')
        model_name: The specific model name/ID

    Returns:
        An instance of the appropriate model class

    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "groq":
        return Groq(id=model_name)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
    
def get_tic_tac_toe_players(
    model_x: str = "groq:llama-3.3-70b-versatile",
    model_o: str = "groq:deepseek-r1-distill-llama-70b",
    debug_mode: bool = True,
) -> Tuple[Agent, Agent]:
    """
    Returns an instance of the Tic Tac Toe Referee Agent that coordinates the game.

    Args:
        model_x: ModelConfig for player X
        model_o: ModelConfig for player O
        model_referee: ModelConfig for the referee agent
        debug_mode: Enable logging and debug features

    Returns:
        An instance of the configured Referee Agent
    """
    
    provider_x, model_name_x = model_x.split(":")
    provider_o, model_name_o = model_o.split(":")
    
    model_x = get_model_provider(provider_x, model_name_x)
    model_o = get_model_provider(provider_o, model_name_o)
    
    player_x = Agent(
        name = "Player X",
        description=dedent("""\
        You are Player X in a Tic Tac Toe game. Your goal is to win by placing three X's in a row (horizontally, vertically, or diagonally).

        BOARD LAYOUT:
        - The board is a 3x3 grid with coordinates from (0,0) to (2,2)
        - Top-left is (0,0), bottom-right is (2,2)

        RULES:
        - You can only place X in empty spaces (shown as " " on the board)
        - Players take turns placing their marks
        - First to get 3 marks in a row (horizontal, vertical, or diagonal) wins
        - If all spaces are filled with no winner, the game is a draw

        YOUR RESPONSE:
        - Provide ONLY two numbers separated by a space (row column)
        - Example: "1 2" places your X in row 1, column 2
        - Choose only from the valid moves list provided to you

        STRATEGY TIPS:
        - Study the board carefully and make strategic moves
        - Block your opponent's potential winning moves
        - Create opportunities for multiple winning paths
        - Pay attention to the valid moves and avoid illegal moves
        """),
        model=model_x,
        retries=3,
        delay_between_retries=30,
        debug_mode=debug_mode
    )
    
    player_o = Agent(
        name = "Player O",
        description=dedent("""\
        You are Player O in a Tic Tac Toe game. Your goal is to win by placing three O's in a row (horizontally, vertically, or diagonally).

        BOARD LAYOUT:
        - The board is a 3x3 grid with coordinates from (0,0) to (2,2)
        - Top-left is (0,0), bottom-right is (2,2)

        RULES:
        - You can only place O in empty spaces (shown as " " on the board)
        - Players take turns placing their marks
        - First to get 3 marks in a row (horizontal, vertical, or diagonal) wins
        - If all spaces are filled with no winner, the game is a draw

        YOUR RESPONSE:
        - Provide ONLY two numbers separated by a space (row column)
        - Example: "1 2" places your O in row 1, column 2
        - Choose only from the valid moves list provided to you

        STRATEGY TIPS:
        - Study the board carefully and make strategic moves
        - Block your opponent's potential winning moves
        - Create opportunities for multiple winning paths
        - Pay attention to the valid moves and avoid illegal moves
        """),
        model=model_o,
        retries=3,
        delay_between_retries=30,
        debug_mode=debug_mode
    )
    
    return player_x, player_o