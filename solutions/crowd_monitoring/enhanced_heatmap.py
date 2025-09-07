import time
import cv2
import numpy as np
import streamlit as st

# Import system path to handle local Ultralytics installation
import sys
sys.path.append("C:/Users/acer/Desktop/new1")

try:
    # Use local Ultralytics installation
    from ultralytics.solutions.heatmap import Heatmap
    from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Local Ultralytics solutions module not available. Falling back to custom implementation.")
    # Import the original implementation as fallback
    from monitor import generate_density_heatmap as fallback_heatmap


class EnhancedHeatmap:
    """
    Enhanced heatmap implementation that leverages Ultralytics solutions.Heatmap()
    while maintaining compatibility with the existing Streamlit UI.
    """
    
    def __init__(self):
        self.heatmaps = {}  # Store heatmap objects per feed_id
        self.last_result = {}  # Store last result per feed_id
        self.frame_counter = {}  # Frame counter per feed_id
        self.update_interval = 3  # Update every N frames
        
        # Check if we can use Ultralytics
        try:
            self.use_ultralytics = ULTRALYTICS_AVAILABLE
            if self.use_ultralytics:
                print("Ultralytics heatmap solution is available and will be used")
            else:
                print("Ultralytics heatmap solution is not available, using fallback implementation")
        except Exception as e:
            print(f"Error checking Ultralytics availability: {e}")
            self.use_ultralytics = False
        
        # Initialize session state for compatibility
        if 'heatmap_accum' not in st.session_state:
            st.session_state.heatmap_accum = {}
        if 'last_heatmap' not in st.session_state:
            st.session_state.last_heatmap = {}
        if not hasattr(st.session_state, 'heatmap_frame_counter'):
            st.session_state.heatmap_frame_counter = {}
    
    def _get_or_create_heatmap(self, feed_id, frame_shape, colormap=cv2.COLORMAP_JET):
        """Get existing heatmap for feed_id or create a new one"""
        if not self.use_ultralytics:
            return None
            
        try:
            if feed_id not in self.heatmaps:
                # Initialize new heatmap for this feed
                self.heatmaps[feed_id] = Heatmap(
                    show=False,  # We'll handle display through Streamlit
                    model=None,  # We'll provide detection results directly
                    colormap=colormap,  # Match existing colormap
                    verbose=False,
                    classes=[0],  # Only track person class
                    decay=0.95,  # Similar to our custom implementation
                    hist_thresh=0.5,  # Lower threshold for better visualization
                )
        except Exception as e:
            print(f"Error initializing Ultralytics Heatmap: {e}")
            return None
            # Initialize the heatmap with a blank frame to set up internal structures
            blank_frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
            self.heatmaps[feed_id].initialized = True  # Ensure initialized flag is set
            
        return self.heatmaps[feed_id]
    
    def generate(self, frame, results, feed_id, colormap=cv2.COLORMAP_JET, decay=0.92):
        """
        Generate a heatmap using Ultralytics solutions.Heatmap while maintaining
        the same interface as the original generate_density_heatmap function.
        
        Args:
            frame: Input frame (BGR format)
            results: Detection results from model
            feed_id: Camera/feed identifier
            colormap: OpenCV colormap to use
            decay: Decay factor for the heatmap (0-1)
            
        Returns:
            Frame with heatmap overlay
        """
        # Store feed_id in session state for compatibility with existing code
        st.session_state["current_feed_id"] = feed_id
        
        # Throttle heatmap updates for performance (update every N frames)
        counter = self.frame_counter.setdefault(feed_id, 0)
        self.frame_counter[feed_id] = (counter + 1) % self.update_interval
        
        # Skip computation and return cached heatmap if not on update frame
        if counter != 0 and feed_id in self.last_result:
            # Update session state for compatibility
            st.session_state.last_heatmap[feed_id] = self.last_result[feed_id]
            return self.last_result[feed_id]
            
        if not self.use_ultralytics:
            # Fall back to original implementation
            return fallback_heatmap(frame, results, colormap, decay)
        
        try:
            # Get or create heatmap for this feed
            heatmap = self._get_or_create_heatmap(feed_id, frame.shape, colormap)
            if heatmap is None:
                # Fall back to original implementation if heatmap creation failed
                return fallback_heatmap(frame, results, colormap, decay)
            
            # Create a copy of the frame to avoid modifying the original
            processed_frame = frame.copy()
            
            # Convert YOLO results to format expected by Ultralytics Heatmap
            boxes = []
            track_ids = []
            classes = []
            
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                for det in results.boxes:
                    # Add more robust error checking
                    if not hasattr(det, 'xyxy') or det.xyxy is None or len(det.xyxy) == 0:
                        continue
                        
                    try:
                        x1, y1, x2, y2 = map(int, det.xyxy[0])
                        cls_id = int(det.cls[0]) if hasattr(det, 'cls') and det.cls is not None and len(det.cls) > 0 else 0
                        
                        if cls_id == 0:  # Only use person class (0)
                            boxes.append([x1, y1, x2, y2])
                            track_ids.append(len(track_ids))  # Generate simple sequential IDs
                            classes.append(cls_id)
                    except (IndexError, TypeError, ValueError) as e:
                        print(f"Skipping detection due to error: {e}")
                
                # Make sure heatmap is valid before proceeding
                if heatmap is None:
                    # Fall back to custom implementation if heatmap initialization failed
                    return fallback_heatmap(frame, results, colormap, decay)
                
                try:
                    # Instead of directly setting properties (which might not be allowed), 
                    # let's create a proper SolutionResults for the heatmap to work with
                    
                    # First make sure our data is valid
                    valid_boxes = boxes if boxes else []
                    valid_track_ids = track_ids if track_ids else []
                    valid_classes = classes if classes else []
                    
                    # Check if we have any detections
                    if valid_boxes and len(valid_boxes) > 0:
                        # Set properties manually but safely
                        if hasattr(heatmap, 'boxes'):
                            heatmap.boxes = valid_boxes
                        if hasattr(heatmap, 'track_ids'):
                            heatmap.track_ids = valid_track_ids
                        if hasattr(heatmap, 'clss'):
                            heatmap.clss = valid_classes
                            
                        # Handle heatmap initialization
                        if not heatmap.initialized:
                            heatmap.heatmap = np.zeros_like(processed_frame, dtype=np.float32) * 0.99
                            heatmap.initialized = True
                        
                        # Process the frame
                        try:
                            heatmap_result = heatmap.process(processed_frame)
                            if hasattr(heatmap_result, 'plot_im') and heatmap_result.plot_im is not None:
                                processed_frame = heatmap_result.plot_im
                            else:
                                # If no visualization, use our custom heatmap implementation
                                processed_frame = fallback_heatmap(frame, results, colormap, decay)
                        except Exception as e:
                            print(f"Error during heatmap processing: {e}")
                            processed_frame = fallback_heatmap(frame, results, colormap, decay)
                    else:
                        # No valid boxes to process, use fallback
                        processed_frame = fallback_heatmap(frame, results, colormap, decay)
                except Exception as e:
                    print(f"Error processing heatmap: {e}")
                    # Fall back to original frame
                    processed_frame = frame.copy()
            else:
                # No detections, use the fallback implementation
                # The fallback implementation handles empty detections better
                processed_frame = fallback_heatmap(frame, results, colormap, decay)
                
            # Store the result
            self.last_result[feed_id] = processed_frame
            
            # Update session state for compatibility
            st.session_state.last_heatmap[feed_id] = processed_frame
            
            return processed_frame
            
        except Exception as e:
            print(f"Error in Ultralytics Heatmap: {e}")
            # Fall back to original implementation
            return fallback_heatmap(frame, results, colormap, decay)


# Create a singleton instance for use throughout the application
enhanced_heatmap = EnhancedHeatmap()


def generate_enhanced_heatmap(frame, results, colormap=cv2.COLORMAP_JET, decay=0.92):
    """
    Drop-in replacement for the original generate_density_heatmap function.
    Uses Ultralytics solutions.Heatmap if available, otherwise falls back to the original.
    
    Args:
        frame: Input frame (BGR format)
        results: Detection results from model
        colormap: OpenCV colormap to use (default: COLORMAP_JET)
        decay: Decay factor for the heatmap (default: 0.92)
        
    Returns:
        Frame with heatmap overlay
    """
    try:
        feed_id = st.session_state.get("current_feed_id", "default")
        if frame is None:
            print("Warning: Received None frame in generate_enhanced_heatmap")
            # Return a blank frame if input is None
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        return enhanced_heatmap.generate(frame, results, feed_id, colormap, decay)
    except Exception as e:
        print(f"Critical error in generate_enhanced_heatmap: {e}")
        # Return the original frame in case of any errors
        if frame is not None:
            return frame.copy()
        return np.zeros((480, 640, 3), dtype=np.uint8)


# For compatibility with existing code - can be imported and used as a direct replacement
generate_density_heatmap = generate_enhanced_heatmap
