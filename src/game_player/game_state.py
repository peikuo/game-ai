#!/usr/bin/env python
"""
Game state objects for structured representation of game state.
Provides type-safe classes for game state handling.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Position in the game."""
    x: int
    y: int


@dataclass
class UIElement:
    """UI element in the game."""
    name: str
    position: List[int]  # [x, y]
    state: str
    function: str


@dataclass
class GameArea:
    """Game area information."""
    board: List[int]  # [x, y, width, height]
    player_area: Optional[List[int]] = None  # [x, y, width, height]
    controls: Optional[List[int]] = None  # [x, y, width, height]
    additional_areas: Optional[Dict[str, List[int]]] = None


@dataclass
class PlayerStatus:
    """Player status information."""
    position: Optional[Union[List[int], str]] = None
    cash: Optional[int] = None
    properties: Optional[List[str]] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class Action:
    """Action to take in the game."""
    type: str
    position: Optional[List[int]] = None
    key: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for GameController."""
        result = {"type": self.type}
        if self.position:
            result["x"] = self.position[0]
            result["y"] = self.position[1]
        if self.key:
            result["key"] = self.key
        if self.parameters:
            result.update(self.parameters)
        return result


@dataclass
class ActionAnalysis:
    """Analysis of what action to take."""
    simple: bool
    explanation: str
    action: Optional[Action] = None


@dataclass
class GameState:
    """Game state information."""
    phase: Optional[str] = None
    current_turn: Optional[str] = None
    player_status: Optional[PlayerStatus] = None
    board_state: Optional[List[Dict[str, Any]]] = None
    resources: Optional[Dict[str, Any]] = None
    other_players: Optional[List[Dict[str, Any]]] = None
    events: Optional[List[Dict[str, Any]]] = None


@dataclass
class GameStateObject:
    """Complete game state object."""
    ui_elements: List[UIElement] = field(default_factory=list)
    game_areas: Optional[GameArea] = None
    game_state: GameState = field(default_factory=GameState)
    action_analysis: ActionAnalysis = field(default_factory=lambda: ActionAnalysis(simple=False, explanation=""))
    raw_description: Optional[str] = None
    monologue: Optional[str] = None
    raw_response: Optional[str] = None


def dict_to_game_state_object(data: Dict[str, Any]) -> GameStateObject:
    """
    Convert a dictionary to a GameStateObject.
    
    Args:
        data: Dictionary representation of game state
        
    Returns:
        GameStateObject instance
    """
    try:
        # Create UI elements
        ui_elements = []
        for ui_data in data.get("ui_elements", []):
            ui_elements.append(UIElement(
                name=ui_data.get("name", ""),
                position=ui_data.get("position", [0, 0]),
                state=ui_data.get("state", "enabled"),
                function=ui_data.get("function", "")
            ))
        
        # Create game areas if present
        game_areas = None
        if "game_areas" in data:
            areas_data = data["game_areas"]
            game_areas = GameArea(
                board=areas_data.get("board", [0, 0, 0, 0]),
                player_area=areas_data.get("player_area"),
                controls=areas_data.get("controls"),
                additional_areas=areas_data.get("additional_areas")
            )
        
        # Create player status if present
        player_status = None
        if "game_state" in data and "player_status" in data["game_state"]:
            status_data = data["game_state"]["player_status"]
            player_status = PlayerStatus(
                position=status_data.get("position"),
                cash=status_data.get("cash"),
                properties=status_data.get("properties"),
                additional_info=status_data.get("additional_info")
            )
        
        # Create game state
        game_state = GameState(
            phase=data.get("game_state", {}).get("phase"),
            current_turn=data.get("game_state", {}).get("current_turn"),
            player_status=player_status,
            board_state=data.get("game_state", {}).get("board_state"),
            resources=data.get("game_state", {}).get("resources"),
            other_players=data.get("game_state", {}).get("other_players"),
            events=data.get("game_state", {}).get("events")
        )
        
        # Create action if present
        action = None
        if "action_analysis" in data and "action" in data["action_analysis"]:
            action_data = data["action_analysis"]["action"]
            if action_data:
                action = Action(
                    type=action_data.get("type", ""),
                    position=action_data.get("position"),
                    key=action_data.get("key"),
                    parameters=action_data.get("parameters")
                )
        
        # Create action analysis
        action_analysis = ActionAnalysis(
            simple=data.get("action_analysis", {}).get("simple", False),
            explanation=data.get("action_analysis", {}).get("explanation", ""),
            action=action
        )
        
        # Create the complete game state object
        return GameStateObject(
            ui_elements=ui_elements,
            game_areas=game_areas,
            game_state=game_state,
            action_analysis=action_analysis,
            raw_description=data.get("raw_description"),
            monologue=data.get("monologue"),
            raw_response=data.get("raw_response")
        )
    
    except Exception as e:
        logger.error(f"Error converting dictionary to GameStateObject: {e}")
        # Return a minimal valid object
        return GameStateObject(
            ui_elements=[],
            game_state=GameState(),
            action_analysis=ActionAnalysis(simple=False, explanation=f"Error: {str(e)}")
        )