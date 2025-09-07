"""
Solution Registry Configuration

This module registers all available solutions in the application.
Import and register all solutions here.
"""
from solutions import register_solution

# Import solution modules - use dynamic loading to avoid import errors
import importlib

# Dynamic import function
def import_module_safely(module_path):
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        print(f"Error importing {module_path}: {e}")
        return None

# Import footfall modules safely
footfall_monitor_bridge = import_module_safely("solutions.footfall.monitor_bridge")
footfall_tasks = import_module_safely("solutions.footfall.tasks") 
footfall_dashboard = import_module_safely("solutions.footfall.dashboard")
footfall_imported = footfall_monitor_bridge is not None

# Get the monitor function from bridge module
footfall_monitor = getattr(footfall_monitor_bridge, "footfall_monitor", None) if footfall_monitor_bridge else None

# Import crowd monitoring modules safely
crowd_monitoring = import_module_safely("solutions.crowd_monitoring")

# Set up function getters for crowd monitoring modules
# This approach avoids direct imports which can cause issues
crowd_monitor = None
crowd_tasks_render = None
crowd_dashboard_render = None

# Set up the module functions if the main module was imported successfully
if crowd_monitoring:
    try:
        crowd_monitor = crowd_monitoring.get_monitor_module()
    except Exception as e:
        print(f"Error getting crowd monitor module: {e}")
        
    try:
        crowd_tasks_render = crowd_monitoring.get_tasks_module()
    except Exception as e:
        print(f"Error getting crowd tasks module: {e}")
        
    try:
        crowd_dashboard_render = crowd_monitoring.get_dashboard_module()
    except Exception as e:
        print(f"Error getting crowd dashboard module: {e}")

# Fallback: Direct imports if the getter functions aren't working
if crowd_monitor is None:
    try:
        crowd_monitor_bridge = import_module_safely("solutions.crowd_monitoring.monitor_bridge")
        crowd_monitor = getattr(crowd_monitor_bridge, "crowd_monitor", None) if crowd_monitor_bridge else None
    except:
        pass

if crowd_tasks_render is None:
    try:
        # First try to import the tasks module directly
        from solutions.crowd_monitoring.tasks import render as direct_tasks_render
        crowd_tasks_render = direct_tasks_render
        print("Successfully loaded crowd tasks through direct import")
    except Exception as e:
        try:
            # If that fails, try the safer import method
            tasks_module = import_module_safely("solutions.crowd_monitoring.tasks")
            crowd_tasks_render = getattr(tasks_module, "render", None) if tasks_module else None
            print("Successfully loaded crowd tasks through safe import")
        except Exception as e:
            print(f"Failed to load crowd tasks: {e}")
            pass
        
if crowd_dashboard_render is None:
    try:
        # First try to import the dashboard module directly
        from solutions.crowd_monitoring.dashboard import render as direct_dashboard_render
        crowd_dashboard_render = direct_dashboard_render
        print("Successfully loaded crowd dashboard through direct import")
    except Exception as e:
        try:
            # If that fails, try the safer import method
            dashboard_module = import_module_safely("solutions.crowd_monitoring.dashboard")
            crowd_dashboard_render = getattr(dashboard_module, "render", None) if dashboard_module else None
            print("Successfully loaded crowd dashboard through safe import")
        except Exception as e:
            print(f"Failed to load crowd dashboard: {e}")
            pass

# Register Footfall solution
if footfall_imported:
    register_solution(
        solution_id="footfall",
        name="Store Footfall & Occupancy",
        description="Count visitors and monitor real-time store occupancy with accurate tracking and analytics",
        task_module=getattr(footfall_tasks, "render", None) if footfall_tasks else None,
        monitor_module=footfall_monitor,
        dashboard_module=getattr(footfall_dashboard, "render", None) if footfall_dashboard else None,
        emoji="👥"
    )

# Register Crowd Monitoring solution
# Always register the crowd monitoring solution even if imports had issues

register_solution(
    solution_id="crowd_monitoring",
    name="Pilgrim Crowd Monitoring",
    description="Monitor crowd density, analyze flow patterns, and get predictive alerts for crowd management",
    task_module=crowd_tasks_render,
    monitor_module=crowd_monitor,
    dashboard_module=crowd_dashboard_render,
    emoji="🛕"
)

# Register future solutions (coming soon)
register_solution(
    solution_id="vision_inspection",
    name="AI Vision Inspection",
    description="Automated quality control and defect detection for manufacturing processes",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="🔍",
    available=False,
    badge="Coming Soon"
)

register_solution(
    solution_id="customer_behavior",
    name="Customer Behavior Analysis",
    description="Track movement patterns, dwell time, and customer engagement in retail environments",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="🛒",
    available=False,
    badge="Coming Soon"
)

register_solution(
    solution_id="product_engagement",
    name="Product Engagement",
    description="Monitor product interactions and shelf performance to optimize product placement",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="📦",
    available=False,
    badge="Coming Soon"
)

register_solution(
    solution_id="service_quality",
    name="Customer Service Quality",
    description="Evaluate service interactions and response times to improve customer experience",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="🛎️",
    available=False,
    badge="Coming Soon"
)

# Register Crowd Monitoring solution
register_solution(
    solution_id="crowd_monitoring",
    name="Pilgrim Crowd Monitoring",
    description="Monitor crowd density, analyze flow patterns, and get predictive alerts for crowd management",
    task_module=crowd_tasks_render,  # Using the function already set up
    monitor_module=crowd_monitor,  # Using the bridge function
    dashboard_module=crowd_dashboard_render,  # Using the function already set up
    emoji="🛕"
)

# Register future solutions (coming soon)
register_solution(
    solution_id="vision_inspection",
    name="AI Vision Inspection",
    description="Automated quality control and defect detection for manufacturing processes",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="🔍",
    available=False,
    badge="Coming Soon"
)

register_solution(
    solution_id="customer_behavior",
    name="Customer Behavior Analysis",
    description="Track movement patterns, dwell time, and customer engagement in retail environments",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="🛒",
    available=False,
    badge="Coming Soon"
)

register_solution(
    solution_id="product_engagement",
    name="Product Engagement",
    description="Monitor product interactions and shelf performance to optimize product placement",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="📦",
    available=False,
    badge="Coming Soon"
)

register_solution(
    solution_id="service_quality",
    name="Customer Service Quality",
    description="Evaluate service interactions and response times to improve customer experience",
    task_module=None,
    monitor_module=None,
    dashboard_module=None,
    emoji="🛎️",
    available=False,
    badge="Coming Soon"
)
