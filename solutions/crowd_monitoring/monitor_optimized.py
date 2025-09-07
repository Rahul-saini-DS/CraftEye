def generate_density_heatmap(frame, results, colormap=cv2.COLORMAP_JET, decay=0.92, stamp_radius=6, blur_ksize=15):
    """
    Optimized incremental heatmap:
    - keeps a per-feed accumulator with decay
    - uses downsampling for performance
    - stamps small blurred blobs at person centers
    """
    H, W = frame.shape[:2]
    feed_id = st.session_state.get("current_feed_id")

    # Throttle heatmap updates for performance (update every N frames)
    if not hasattr(st.session_state, 'heatmap_frame_counter'):
        st.session_state.heatmap_frame_counter = {}
    
    # Initialize feed-specific counters and last timestamps
    feed_counter = st.session_state.heatmap_frame_counter.setdefault(feed_id, 0)
    st.session_state.heatmap_frame_counter[feed_id] = (feed_counter + 1) % 3  # Update every 3 frames
    
    if feed_counter != 0 and hasattr(st.session_state, 'last_heatmap') and feed_id in st.session_state.last_heatmap:
        # Skip computation and return cached heatmap
        return st.session_state.last_heatmap[feed_id]
        
    # Initialize accumulators
    if 'heatmap_accum' not in st.session_state:
        st.session_state.heatmap_accum = {}
    if 'last_heatmap' not in st.session_state:
        st.session_state.last_heatmap = {}
        
    if feed_id not in st.session_state.heatmap_accum:
        st.session_state.heatmap_accum[feed_id] = np.zeros((H, W), dtype=np.float32)

    accum = st.session_state.heatmap_accum[feed_id]
    # decay previous frame's energy
    cv2.multiply(accum, decay, dst=accum)

    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        # Use downsampling for better performance
        small_scale = 0.5
        h2, w2 = int(H * small_scale), int(W * small_scale)
        
        # Create stamp on smaller canvas
        stamp = np.zeros((h2, w2), dtype=np.uint8)
        
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            # Scale coordinates to smaller size
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx_small = int(cx * small_scale)
            cy_small = int(cy * small_scale)
            cv2.circle(stamp, (cx_small, cy_small), stamp_radius, 255, -1)
            
        # Blur at smaller resolution (much faster)
        stamp = cv2.GaussianBlur(stamp, (blur_ksize, blur_ksize), 0)
        
        # Scale back to original size
        stamp = cv2.resize(stamp, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Add to accumulator
        inc = (stamp.astype(np.float32) / 255.0) * 0.6
        cv2.add(accum, inc, dst=accum, dtype=cv2.CV_32F)
    
    # Cache the result
    result = _overlay_from_accum(frame, accum, colormap)
    st.session_state.last_heatmap[feed_id] = result
    return result
