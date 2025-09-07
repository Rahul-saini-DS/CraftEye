# CraftEye: Crowd Monitoring System for Simhastha Maha Kumbh

![CraftEye Logo](assets/CraftEye%20LOGO.png)

## 📋 Overview

CraftEye is an advanced AI-powered crowd monitoring platform specifically optimized for the Simhastha Maha Kumbh festival. This vision intelligence system provides real-time crowd density analysis, flow tracking, and occupancy monitoring to ensure safety and enhance management during one of the world's largest religious gatherings.

## 🔗 Public Code Repository

[https://github.com/Rahul-saini-DS/CraftEye](https://github.com/Rahul-saini-DS/CraftEye)

## 🌟 Key Features

- **Real-time Crowd Density Analysis**: Monitor crowd densities across different zones of the Maha Kumbh venue in real-time  
- **Entry/Exit Tracking**: Count pilgrims entering and exiting designated areas with precise flow tracking  
- **Zone-based Occupancy Management**: Define safety thresholds for different zones and receive alerts when thresholds are exceeded  
- **Interactive Analytics Dashboard**: Visualize crowd patterns, peak times, and historical data  
- **Multi-camera Support**: Seamless integration with existing CCTV infrastructure  
- **Scalable Architecture**: Designed to handle the massive scale of Simhastha Maha Kumbh (millions of attendees)  

## 🎯 Core Problem Addressed

The Simhastha Maha Kumbh festival faces critical challenges in crowd management and safety:

1. **Overcrowding Risks**: Dangerous crowd densities can develop rapidly in key areas  
2. **Limited Visibility**: Traditional monitoring systems can't provide comprehensive real-time insights  
3. **Resource Allocation**: Security and medical teams need data-driven deployment strategies  
4. **Flow Management**: Understanding crowd movement patterns to prevent bottlenecks  
5. **Emergency Response**: Quick identification of potentially hazardous situations  

CraftEye solves these challenges by providing authorities with an AI-powered "eye in the sky" that continuously monitors crowd conditions, predicts potential issues, and enables proactive management.

## 🛠️ Technology Stack

- **Computer Vision**: YOLO-based person detection and tracking  
- **Data Visualization**: Interactive Streamlit dashboard with Plotly graphs  
- **Backend**: Optimized Python pipeline for real-time processing  
- **Deployment**: Containerized for easy scaling across multiple monitoring stations  

## 📊 Demonstration Scenario

CraftEye has been configured to monitor key areas of the Simhastha Maha Kumbh:

1. **Main Bathing Ghats**: Track occupancy levels at river banks during peak ceremony times  
2. **Temple Entrances**: Manage queue densities at major temples and shrines  
3. **Transit Corridors**: Monitor flow rates in key pedestrian pathways  
4. **Emergency Exit Routes**: Ensure evacuation routes remain clear and accessible  

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Rahul-saini-DS/CraftEye.git
cd CraftEye

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run Home.py
