"""
Predictive Analytics Module for Crowd Monitoring

This module provides predictive analytics capabilities for crowd monitoring,
including trend analysis, anomaly detection, and early warning alerts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Any, Optional
import time
import streamlit as st

class PredictiveAnalytics:
    """
    Class for performing predictive analytics on crowd monitoring data.
    """
    
    def __init__(self):
        """Initialize the predictive analytics engine."""
        self.density_history = []
        self.timestamps = []
        self.last_predictions = {}
        self.alert_thresholds = {
            'warning': 0.7,  # people per m²
            'critical': 0.9   # people per m²
        }
        
    def add_data_point(self, density: float, timestamp: Optional[float] = None) -> None:
        """
        Add a new density data point for analysis.
        
        Args:
            density: Crowd density value (people per m²)
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.density_history.append(density)
        self.timestamps.append(timestamp)
        
        # Keep a reasonable history (last 2 hours)
        if len(self.density_history) > 1440:  # 2 hours at 1 sample per 5 seconds
            self.density_history = self.density_history[-1440:]
            self.timestamps = self.timestamps[-1440:]
    
    def predict_future_density(self, minutes_ahead: int = 15) -> Dict[str, Any]:
        """
        Predict future density based on recent trend.
        
        Args:
            minutes_ahead: Number of minutes to predict ahead
            
        Returns:
            Dictionary with prediction results
        """
        # Need at least 10 data points for prediction
        if len(self.density_history) < 10:
            return {
                'status': 'insufficient_data',
                'message': 'Insufficient data for prediction (need at least 10 data points)',
                'predictions': [],
                'timestamps': []
            }
        
        try:
            # Use recent history (last 20 points or all if fewer)
            recent_history = self.density_history[-20:]
            recent_timestamps = self.timestamps[-20:]
            
            # Simple linear regression for prediction
            x = np.arange(len(recent_history)).reshape(-1, 1)
            y = np.array(recent_history)
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(x, y)
            
            # Generate future timestamps and predictions
            future_timestamps = []
            last_time = recent_timestamps[-1]
            for i in range(1, minutes_ahead + 1):
                future_timestamps.append(last_time + (i * 60))  # Add i minutes in seconds
            
            # Predict future values
            future_x = np.arange(len(recent_history), len(recent_history) + minutes_ahead).reshape(-1, 1)
            future_y = model.predict(future_x)
            
            # Ensure predictions are non-negative
            future_y = np.maximum(future_y, 0)
            
            # Store prediction for later comparison
            self.last_predictions = {
                'timestamps': future_timestamps,
                'values': future_y.tolist(),
                'generated_at': time.time()
            }
            
            # Detect if predictions cross thresholds
            current_max = max(recent_history)
            warning_crossed = False
            critical_crossed = False
            warning_time = None
            critical_time = None
            
            # Check if predictions cross thresholds
            for i, pred in enumerate(future_y):
                if current_max < self.alert_thresholds['warning'] and pred >= self.alert_thresholds['warning']:
                    warning_crossed = True
                    warning_time = i + 1  # Minutes ahead
                
                if current_max < self.alert_thresholds['critical'] and pred >= self.alert_thresholds['critical']:
                    critical_crossed = True
                    critical_time = i + 1  # Minutes ahead
            
            alerts = []
            if critical_crossed:
                alerts.append({
                    'level': 'critical',
                    'message': f'🚨 CRITICAL: Projected to reach critical density in ~{critical_time} minutes',
                    'time_to_threshold': critical_time
                })
            elif warning_crossed:
                alerts.append({
                    'level': 'warning',
                    'message': f'⚠️ WARNING: Projected to reach high density in ~{warning_time} minutes',
                    'time_to_threshold': warning_time
                })
            
            return {
                'status': 'success',
                'predictions': future_y.tolist(),
                'timestamps': future_timestamps,
                'alerts': alerts,
                'model_slope': float(model.coef_[0]),
                'trend': 'increasing' if model.coef_[0] > 0.01 else 'decreasing' if model.coef_[0] < -0.01 else 'stable'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Prediction error: {str(e)}',
                'predictions': [],
                'timestamps': []
            }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in crowd density data.
        
        Returns:
            List of anomaly events
        """
        # Need at least 30 data points for anomaly detection
        if len(self.density_history) < 30:
            return []
        
        try:
            # Calculate rolling mean and standard deviation
            density_array = np.array(self.density_history)
            window_size = min(20, len(density_array) // 2)
            
            # Simple rolling statistics
            rolling_mean = np.convolve(density_array, np.ones(window_size)/window_size, mode='valid')
            
            # Extend rolling mean to match original array length
            padding = len(density_array) - len(rolling_mean)
            extended_mean = np.pad(rolling_mean, (padding, 0), 'edge')
            
            # Calculate deviations
            deviations = np.abs(density_array - extended_mean)
            
            # Threshold for anomaly detection (2 standard deviations)
            threshold = 2 * np.std(deviations)
            
            # Find anomalies (focus on recent data)
            anomalies = []
            lookback = min(30, len(density_array))  # Look at last 30 points max
            
            for i in range(len(density_array) - lookback, len(density_array)):
                if deviations[i] > threshold:
                    anomalies.append({
                        'timestamp': self.timestamps[i],
                        'density': self.density_history[i],
                        'expected': extended_mean[i],
                        'deviation': deviations[i]
                    })
            
            return anomalies
            
        except Exception as e:
            print(f"Anomaly detection error: {str(e)}")
            return []
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive trend analysis of crowd density.
        
        Returns:
            Dictionary with trend analysis results
        """
        if len(self.density_history) < 5:
            return {
                'status': 'insufficient_data',
                'trend': 'unknown',
                'message': 'Insufficient data for trend analysis'
            }
        
        # Use recent data for trend
        recent = self.density_history[-5:]
        
        # Calculate simple trend
        slope = (recent[-1] - recent[0]) / 5 if len(recent) >= 5 else 0
        
        # Determine trend direction
        if slope > 0.05:
            trend = 'increasing'
            message = 'Crowd density is increasing'
        elif slope < -0.05:
            trend = 'decreasing' 
            message = 'Crowd density is decreasing'
        else:
            trend = 'stable'
            message = 'Crowd density is stable'
        
        # Get latest density
        current_density = self.density_history[-1] if self.density_history else 0
        
        # Get status based on thresholds
        if current_density >= self.alert_thresholds['critical']:
            status = 'critical'
        elif current_density >= self.alert_thresholds['warning']:
            status = 'warning'
        else:
            status = 'normal'
        
        # Future prediction
        prediction = self.predict_future_density(15)
        
        return {
            'status': 'success',
            'trend': trend,
            'message': message,
            'current_density': current_density,
            'density_status': status,
            'slope': slope,
            'prediction': prediction
        }

# Function to create streamlit UI for the predictive analytics
def display_predictive_analytics(feed_id: str = None) -> None:
    """
    Create a Streamlit UI to display predictive analytics.
    
    Args:
        feed_id: Optional camera feed ID to show analytics for
    """
    # Initialize analytics engine if not exists
    if 'predictive_analytics' not in st.session_state:
        st.session_state['predictive_analytics'] = {}
    
    # Create engine for this feed if needed
    if feed_id and feed_id not in st.session_state['predictive_analytics']:
        st.session_state['predictive_analytics'][feed_id] = PredictiveAnalytics()
    
    # Get analytics for this feed
    analytics = st.session_state['predictive_analytics'].get(feed_id) if feed_id else None
    
    st.subheader("Crowd Density Prediction")
    
    if not analytics or len(analytics.density_history) < 10:
        st.info("Collecting data for prediction... (need at least 10 data points)")
        return
    
    # Run prediction
    prediction_results = analytics.predict_future_density(15)
    
    if prediction_results['status'] != 'success':
        st.warning(prediction_results['message'])
        return
    
    # Display alerts if any
    for alert in prediction_results.get('alerts', []):
        if alert['level'] == 'critical':
            st.error(alert['message'])
        elif alert['level'] == 'warning':
            st.warning(alert['message'])
    
    # Display trend
    trend = prediction_results['trend']
    trend_icon = "📈" if trend == "increasing" else "📉" if trend == "decreasing" else "📊"
    st.info(f"{trend_icon} Density trend: {trend.title()}")
    
    # Create prediction chart
    import plotly.graph_objects as go
    from datetime import datetime
    
    # Historical data
    historical_times = [datetime.fromtimestamp(ts) for ts in analytics.timestamps[-20:]]
    historical_values = analytics.density_history[-20:]
    
    # Prediction data
    future_times = [datetime.fromtimestamp(ts) for ts in prediction_results['timestamps']]
    future_values = prediction_results['predictions']
    
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_times,
        y=historical_values,
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add prediction
    fig.add_trace(go.Scatter(
        x=future_times,
        y=future_values,
        mode='lines+markers',
        name='Prediction',
        line=dict(color='purple', dash='dash')
    ))
    
    # Add warning threshold line
    fig.add_shape(
        type="line",
        x0=historical_times[0],
        y0=analytics.alert_thresholds['warning'],
        x1=future_times[-1],
        y1=analytics.alert_thresholds['warning'],
        line=dict(color="orange", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=future_times[-1],
        y=analytics.alert_thresholds['warning'],
        text="Warning Threshold",
        showarrow=False,
        xshift=10,
        font=dict(color="orange")
    )
    
    # Add critical threshold line
    fig.add_shape(
        type="line",
        x0=historical_times[0],
        y0=analytics.alert_thresholds['critical'],
        x1=future_times[-1],
        y1=analytics.alert_thresholds['critical'],
        line=dict(color="red", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=future_times[-1],
        y=analytics.alert_thresholds['critical'],
        text="Critical Threshold",
        showarrow=False,
        xshift=10,
        font=dict(color="red")
    )
    
    fig.update_layout(
        title="Crowd Density Prediction (Next 15 Minutes)",
        xaxis_title="Time",
        yaxis_title="Density (people/m²)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Function to update the predictive analytics with new data
def update_analytics(feed_id: str, density: float, timestamp: Optional[float] = None) -> None:
    """
    Update analytics for a specific feed with new data.
    
    Args:
        feed_id: Camera feed ID
        density: Current density value
        timestamp: Optional timestamp (defaults to current time)
    """
    if 'predictive_analytics' not in st.session_state:
        st.session_state['predictive_analytics'] = {}
    
    if feed_id not in st.session_state['predictive_analytics']:
        st.session_state['predictive_analytics'][feed_id] = PredictiveAnalytics()
    
    analytics = st.session_state['predictive_analytics'][feed_id]
    analytics.add_data_point(density, timestamp)
    
    # Check for alerts based on predictions
    if len(analytics.density_history) >= 10:
        prediction = analytics.predict_future_density(15)
        
        # Process alerts if needed
        for alert in prediction.get('alerts', []):
            # Add to global alerts list
            if 'alerts' not in st.session_state:
                st.session_state['alerts'] = []
            
            # Check if this is a new alert
            alert_msg = alert['message']
            if not any(alert_msg in a for a in st.session_state['alerts']):
                st.session_state['alerts'].insert(0, alert_msg)
                if len(st.session_state['alerts']) > 10:
                    st.session_state['alerts'] = st.session_state['alerts'][:10]
