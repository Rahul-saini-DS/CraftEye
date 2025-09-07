"""
Crowd Monitoring solution package

This solution provides crowd density analysis and monitoring.
"""

# Define package metadata
__solution_id__ = "crowd_monitoring"
__solution_name__ = "Pilgrim Crowd Monitoring"
__solution_description__ = "Monitor crowd density, analyze flow patterns, and get predictive alerts for crowd management"

# Make key functions available for direct import
# Note: Using functions to avoid immediate imports that might cause circular imports
def get_monitor_module():
    from solutions.crowd_monitoring.monitor_bridge import crowd_monitor
    return crowd_monitor

def get_dashboard_module():
    from solutions.crowd_monitoring.dashboard import render
    return render

def get_tasks_module():
    from solutions.crowd_monitoring.tasks import render
    return render
