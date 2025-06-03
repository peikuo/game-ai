#!/usr/bin/env python
"""
Game state object module for structured access to game state data.
Provides typed classes for game state components.
"""

from typing import Any, Dict, List, Optional, Union


class Position:
    """Position in the game (x, y coordinates)"""

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y})"


class UIElement:
    """UI element in the game (button, text field, etc.)"""

    def __init__(self, element_id: str = "", element_type: str = "",
                 text: str = "", position: Optional[Position] = None,
                 clickable: bool = False, enabled: bool = True):
        self.element_id = element_id
        self.element_type = element_type
        self.text = text
        self.position = position or Position()
        self.clickable = clickable
        self.enabled = enabled

    def __repr__(self) -> str:
        return f"UIElement(id={self.element_id}, type={self.element_type}, text={self.text})"


class Action:
    """Game action to be executed"""

    def __init__(self, action_type: str = "", position: Optional[Position] = None,
                 key: Optional[str] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        self.type = action_type
        self.position = position
        self.key = key
        self.parameters = parameters or {}

    def __repr__(self) -> str:
        return f"Action(type={self.type}, position={self.position}, key={self.key})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for the game controller"""
        result = {"type": self.type}
    
        if self.position:
            # Add both position array and individual x,y coordinates
            # for backward compatibility
            result["position"] = [self.position.x, self.position.y]
            result["x"] = self.position.x
            result["y"] = self.position.y
    
        if self.key:
            result["key"] = self.key
        
        if self.parameters:
            result["parameters"] = self.parameters
        
        return result


class ActionAnalysis:
    """Analysis of what action to take"""

    def __init__(self, simple: bool = False, explanation: str = "",
                 action: Optional[Action] = None):
        self.simple = simple
        self.explanation = explanation
        self.action = action

    def __repr__(self) -> str:
        return f"ActionAnalysis(simple={self.simple}, action={self.action})"


class PlayerStatus:
    """Player status in the game"""

    def __init__(self, name: str = "", score: Union[int, str] = 0,
                 resources: Optional[Dict[str, Any]] = None):
        self.name = name
        self.score = score
        self.resources = resources or {}

    def __repr__(self) -> str:
        return f"PlayerStatus(name={self.name}, score={self.score})"


class GameArea:
    """Game areas identified in the screenshot"""

    def __init__(self, main_area: Optional[Dict[str, Any]] = None,
                 menu_area: Optional[Dict[str, Any]] = None,
                 status_area: Optional[Dict[str, Any]] = None):
        self.main_area = main_area or {}
        self.menu_area = menu_area or {}
        self.status_area = status_area or {}

    def __repr__(self) -> str:
        return f"GameArea(main={bool(self.main_area)}, menu={bool(self.menu_area)}, status={bool(self.status_area)})"


class GameState:
    """Game state information"""

    def __init__(self, phase: str = "", current_turn: str = "",
                 player_status: Optional[PlayerStatus] = None,
                 board_state: Optional[List[Dict[str, Any]]] = None,
                 resources: Optional[Dict[str, Any]] = None,
                 other_players: Optional[List[Dict[str, Any]]] = None,
                 events: Optional[List[Dict[str, Any]]] = None):
        self.phase = phase
        self.current_turn = current_turn
        self.player_status = player_status or PlayerStatus()
        self.board_state = board_state or []
        self.resources = resources or {}
        self.other_players = other_players or []
        self.events = events or []

    def __repr__(self) -> str:
        return f"GameState(phase={self.phase}, turn={self.current_turn})"


class GameStateObject:
    """
    Main game state object that contains all analysis results.
    This is the top-level object returned by the image analyzer.
    """

    def __init__(self, ui_elements: Optional[List[UIElement]] = None,
                 game_areas: Optional[GameArea] = None,
                 game_state: Optional[GameState] = None,
                 action_analysis: Optional[ActionAnalysis] = None,
                 raw_description: str = "",
                 monologue: str = "",
                 raw_response: str = ""):
        self.ui_elements = ui_elements or []
        self.game_areas = game_areas or GameArea()
        self.game_state = game_state or GameState()
        self.action_analysis = action_analysis or ActionAnalysis()
        self.raw_description = raw_description
        self.monologue = monologue
        self.raw_response = raw_response

    def __repr__(self) -> str:
        return (f"GameStateObject(elements={len(self.ui_elements)}, "
                f"action={self.action_analysis})")


def dict_to_game_state_object(data: Dict[str, Any]) -> GameStateObject:
    """
    Convert a dictionary to a GameStateObject.

    Args:
        data: Dictionary from the image analyzer
    
    Returns:
        GameStateObject with all components properly initialized
    """
    # Create UI elements
    ui_elements = []
    for elem_data in data.get("ui_elements", []):
        position = None
        if "position" in elem_data:
            pos = elem_data["position"]
            if isinstance(pos, list) and len(pos) >= 2:
                position = Position(pos[0], pos[1])
    
        ui_elements.append(UIElement(
            element_id=elem_data.get("id", ""),
            element_type=elem_data.get("type", ""),
            text=elem_data.get("text", ""),
            position=position,
            clickable=elem_data.get("clickable", False),
            enabled=elem_data.get("enabled", True)
        ))

    # Create action
    action = None
    action_data = data.get("action_analysis", {}).get("action")
    if action_data:
        position = None
        if "position" in action_data:
            pos = action_data["position"]
            if isinstance(pos, list) and len(pos) >= 2:
                position = Position(pos[0], pos[1])
    
        action = Action(
            action_type=action_data.get("type", ""),
            position=position,
            key=action_data.get("key"),
            parameters=action_data.get("parameters", {})
        )

    # Create action analysis
    action_analysis = ActionAnalysis(
        simple=data.get("action_analysis", {}).get("simple", False),
        explanation=data.get("action_analysis", {}).get("explanation", ""),
        action=action
    )

    # Create player status
    player_status = None
    player_data = data.get("game_state", {}).get("player_status")
    if player_data:
        player_status = PlayerStatus(
            name=player_data.get("name", ""),
            score=player_data.get("score", 0),
            resources=player_data.get("resources", {})
        )

    # Create game areas
    game_areas = GameArea(
        main_area=data.get("game_areas", {}).get("main_area", {}),
        menu_area=data.get("game_areas", {}).get("menu_area", {}),
        status_area=data.get("game_areas", {}).get("status_area", {})
    )

    # Create game state
    game_state = GameState(
        phase=data.get("game_state", {}).get("phase", ""),
        current_turn=data.get("game_state", {}).get("current_turn", ""),
        player_status=player_status,
        board_state=data.get("game_state", {}).get("board_state", []),
        resources=data.get("game_state", {}).get("resources", {}),
        other_players=data.get("game_state", {}).get("other_players", []),
        events=data.get("game_state", {}).get("events", [])
    )

    # Create the main game state object
    return GameStateObject(
        ui_elements=ui_elements,
        game_areas=game_areas,
        game_state=game_state,
        action_analysis=action_analysis,
        raw_description=data.get("raw_description", ""),
        monologue=data.get("monologue", ""),
        raw_response=data.get("raw_response", "")
    )
