"""
Solution Registry System

This module maintains a registry of all available solutions in the application.
Each solution must register its task, monitor, and dashboard modules.
"""
from typing import Dict, List, Any, Callable, Optional

# Solution registry
SOLUTIONS = {}

def register_solution(
    solution_id: str, 
    name: str, 
    description: str,
    task_module: Optional[Callable] = None, 
    monitor_module: Optional[Callable] = None,
    dashboard_module: Optional[Callable] = None,
    emoji: str = "📊",
    badge: str = None,
    available: bool = True
):
    """Register a solution in the global registry.
    
    Args:
        solution_id: Unique identifier for the solution
        name: Display name of the solution
        description: Brief description of what the solution does
        task_module: Function to render the task configuration UI
        monitor_module: Function to render the monitoring UI
        dashboard_module: Function to render the dashboard UI
        emoji: Icon to display alongside the solution
        badge: Optional badge text (e.g., "New", "Beta")
        available: Whether the solution is currently available
    """
    SOLUTIONS[solution_id] = {
        "id": solution_id,
        "name": name,
        "description": description,
        "task_module": task_module,
        "monitor_module": monitor_module,
        "dashboard_module": dashboard_module,
        "emoji": emoji,
        "badge": badge,
        "available": available
    }

def get_solution(solution_id: str) -> Dict[str, Any]:
    """Get a solution by ID."""
    return SOLUTIONS.get(solution_id)

def get_solution_by_name(name: str) -> Dict[str, Any]:
    """Get a solution by name."""
    # First try exact match
    for solution in SOLUTIONS.values():
        if solution["name"] == name:
            return solution
    
    # If not found, try case-insensitive partial match
    name_lower = name.lower()
    for solution in SOLUTIONS.values():
        if name_lower in solution["name"].lower():
            return solution
            
    # Special case for Pilgrim Crowd Monitoring solution
    # (ensure it's always available for dashboard and monitor pages)
    if "pilgrim" in name_lower or "crowd" in name_lower:
        for solution in SOLUTIONS.values():
            if "pilgrim" in solution["name"].lower() or "crowd" in solution["name"].lower():
                return solution
                
    return None

def list_available_solutions() -> List[Dict[str, Any]]:
    """List all available solutions."""
    return [
        {
            "id": sol_id,
            "name": sol["name"],
            "description": sol["description"],
            "emoji": sol["emoji"],
            "badge": sol["badge"]
        }
        for sol_id, sol in SOLUTIONS.items()
        if sol["available"]
    ]

def list_coming_soon_solutions() -> List[Dict[str, Any]]:
    """List all coming soon solutions."""
    return [
        {
            "id": sol_id,
            "name": sol["name"],
            "description": sol["description"],
            "emoji": sol["emoji"],
            "badge": sol["badge"] or "Coming Soon"
        }
        for sol_id, sol in SOLUTIONS.items()
        if not sol["available"]
    ]
