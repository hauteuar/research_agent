# Enhanced GPU Status Functions for Streamlit - Part 1
# Add these functions to your streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import asyncio
from typing import Dict, Any, List
import sqlite3  # Should already be there
import os

def get_detailed_server_status():
    """Get detailed status from all model servers including GPU information"""
    if not st.session_state.coordinator:
        return {}
    
    detailed_status = {}
    
    try:
        # Get coordinator health which includes server stats
        health = st.session_state.coordinator.get_health_status()
        server_stats = health.get('server_stats', {})
        
        # For each server, try to get detailed GPU information
        for server_name, basic_stats in server_stats.items():
            try:
                # Find the server configuration to get endpoint
                server_config = None
                for server in st.session_state.model_servers:
                    if server.get('name') == server_name:
                        server_config = server
                        break
                
                if server_config and server_config.get('endpoint'):
                    # Get detailed status from the model server directly
                    response = requests.get(
                        f"{server_config['endpoint']}/status",
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        server_status = response.json()
                        
                        # Merge basic stats with detailed server info
                        detailed_status[server_name] = {
                            **basic_stats,
                            'endpoint': server_config['endpoint'],
                            'gpu_info': server_status.get('gpu_info', {}),
                            'memory_info': server_status.get('memory_info', {}),
                            'model': server_status.get('model', 'Unknown'),
                            'uptime': server_status.get('uptime', 0)
                        }
                    else:
                        # Fallback to basic stats
                        detailed_status[server_name] = {
                            **basic_stats,
                            'endpoint': server_config['endpoint'],
                            'error': f"HTTP {response.status_code}"
                        }
                else:
                    # No endpoint configuration found
                    detailed_status[server_name] = {
                        **basic_stats,
                        'error': 'No endpoint configured'
                    }
                    
            except requests.RequestException as e:
                # Server unreachable
                detailed_status[server_name] = {
                    **basic_stats,
                    'error': f"Connection error: {str(e)}"
                }
            except Exception as e:
                # Other errors
                detailed_status[server_name] = {
                    **basic_stats,
                    'error': f"Error: {str(e)}"
                }
        
        return detailed_status
        
    except Exception as e:
        st.error(f"Failed to get detailed server status: {str(e)}")
        return {}

def show_enhanced_server_status():
    """Enhanced server status display with GPU information"""
    if st.session_state.coordinator:
        detailed_status = get_detailed_server_status()
        
        if not detailed_status:
            st.warning("No server status available")
            return
        
        # Overall system health
        healthy_servers = sum(1 for stats in detailed_status.values() 
                            if stats.get('status') == 'healthy' and 'error' not in stats)
        total_servers = len(detailed_status)
        
        # Main status indicator
        if healthy_servers == total_servers:
            st.success(f"🟢 All Systems Operational - {healthy_servers}/{total_servers} servers healthy")
        elif healthy_servers > 0:
            st.warning(f"⚠️ Partial Service - {healthy_servers}/{total_servers} servers healthy")
        else:
            st.error(f"🔴 Service Degraded - {healthy_servers}/{total_servers} servers healthy")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🖥️ Server Overview", "🎮 GPU Details", "📊 Performance"])
        
        with tab1:
            show_server_overview_tab(detailed_status)
        
        with tab2:
            show_gpu_details_tab(detailed_status)
        
        with tab3:
            show_performance_tab(detailed_status)
    else:
        st.warning("🟡 API Coordinator not initialized")

def show_server_overview_tab(detailed_status: Dict):
    """Show server overview with basic health and performance"""
    for server_name, stats in detailed_status.items():
        with st.expander(f"🖥️ {server_name} - {'🟢' if stats.get('status') == 'healthy' else '🔴'}", expanded=True):
            
            if 'error' in stats:
                st.error(f"❌ {stats['error']}")
                st.caption(f"Endpoint: {stats.get('endpoint', 'Unknown')}")
                continue
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                status = stats.get('status', 'unknown')
                if status == 'healthy':
                    st.success("🟢 Healthy")
                else:
                    st.error(f"🔴 {status}")
            
            with col2:
                active_requests = stats.get('active_requests', 0)
                st.metric("Active Requests", active_requests)
            
            with col3:
                total_requests = stats.get('total_requests', 0)
                st.metric("Total Requests", total_requests)
            
            with col4:
                success_rate = stats.get('success_rate', 0)
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Additional server info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Endpoint:** {stats.get('endpoint', 'Unknown')}")
            
            with col2:
                model = stats.get('model', 'Unknown')
                st.info(f"**Model:** {model}")
            
            with col3:
                uptime = stats.get('uptime', 0)
                st.info(f"**Uptime:** {uptime:.0f}s")

def show_gpu_details_tab(detailed_status: Dict):
    """Show detailed GPU information for each server"""
    st.markdown("### 🎮 GPU Status & Allocation")
    
    # Collect GPU data for visualization
    gpu_data = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
            
        gpu_info = stats.get('gpu_info', {})
        
        if not gpu_info:
            st.warning(f"No GPU information available for {server_name}")
            continue
        
        # Display GPU details for this server
        with st.expander(f"🎮 {server_name} GPU Details", expanded=True):
            
            # Create columns for each GPU
            gpu_names = list(gpu_info.keys())
            if len(gpu_names) == 1:
                cols = [st.container()]
            else:
                cols = st.columns(len(gpu_names))
            
            for i, (gpu_name, gpu_data_item) in enumerate(gpu_info.items()):
                col = cols[i] if len(gpu_names) > 1 else cols[0]
                
                with col:
                    st.markdown(f"#### {gpu_name}")
                    
                    # GPU basic info
                    gpu_name_str = gpu_data_item.get('name', 'Unknown GPU')
                    compute_cap = gpu_data_item.get('compute_capability', 'Unknown')
                    st.info(f"**GPU:** {gpu_name_str}")
                    st.info(f"**Compute:** {compute_cap}")
                    
                    # Memory information
                    total_memory = gpu_data_item.get('total_memory', 0)
                    memory_allocated = gpu_data_item.get('memory_allocated', 0)
                    memory_cached = gpu_data_item.get('memory_cached', 0)
                    memory_free = gpu_data_item.get('memory_free', 0)
                    utilization = gpu_data_item.get('utilization', 0)
                    
                    # Memory metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if total_memory > 0:
                            allocated_gb = memory_allocated / (1024**3)
                            total_gb = total_memory / (1024**3)
                            st.metric("Memory Allocated", f"{allocated_gb:.1f}GB / {total_gb:.1f}GB")
                        else:
                            st.metric("Memory Allocated", "Unknown")
                    
                    with col2:
                        st.metric("GPU Utilization", f"{utilization:.1f}%")
                    
                    # Memory usage progress bar
                    if total_memory > 0:
                        memory_usage_percent = (memory_cached / total_memory) * 100
                        st.progress(memory_usage_percent / 100)
                        st.caption(f"Memory Usage: {memory_usage_percent:.1f}%")
                    
                    # Collect data for charts
                    gpu_data.append({
                        'server': server_name,
                        'gpu': gpu_name,
                        'utilization': utilization,
                        'memory_used_gb': memory_cached / (1024**3) if total_memory > 0 else 0,
                        'memory_total_gb': total_memory / (1024**3) if total_memory > 0 else 0,
                        'memory_percent': (memory_cached / total_memory) * 100 if total_memory > 0 else 0
                    })
    
    # Show GPU utilization charts
    if gpu_data:
        st.markdown("### 📊 GPU Utilization Overview")
        
        df_gpu = pd.DataFrame(gpu_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # GPU Utilization by Server
            fig_util = px.bar(df_gpu, x="server", y="utilization", 
                            title="GPU Utilization by Server (%)",
                            color="utilization",
                            color_continuous_scale="RdYlGn_r")
            fig_util.update_layout(height=300)
            st.plotly_chart(fig_util, use_container_width=True)
        
        with col2:
            # GPU Memory Usage by Server  
            fig_mem = px.bar(df_gpu, x="server", y="memory_percent",
                           title="GPU Memory Usage by Server (%)",
                           color="memory_percent", 
                           color_continuous_scale="RdYlBu_r")
            fig_mem.update_layout(height=300)
            st.plotly_chart(fig_mem, use_container_width=True)
    # Enhanced GPU Status Functions for Streamlit - Part 2
# Add these functions to your streamlit_app.py

def show_performance_tab(detailed_status: Dict):
    """Show detailed performance metrics including GPU performance"""
    st.markdown("### 📊 Performance Analytics")
    
    # Collect performance data
    perf_data = []
    server_summary = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
        
        # Server performance summary
        server_summary.append({
            'server': server_name,
            'status': stats.get('status', 'unknown'),
            'active_requests': stats.get('active_requests', 0),
            'total_requests': stats.get('total_requests', 0),
            'success_rate': stats.get('success_rate', 0),
            'avg_latency': stats.get('average_latency', 0),
            'uptime': stats.get('uptime', 0)
        })
        
        # GPU performance data
        gpu_info = stats.get('gpu_info', {})
        for gpu_name, gpu_data in gpu_info.items():
            perf_data.append({
                'server': server_name,
                'gpu': gpu_name,
                'utilization': gpu_data.get('utilization', 0),
                'memory_used': gpu_data.get('memory_cached', 0) / (1024**3),
                'memory_total': gpu_data.get('total_memory', 0) / (1024**3),
                'active_requests': stats.get('active_requests', 0)
            })
    
    if server_summary:
        df_servers = pd.DataFrame(server_summary)
        
        # Performance overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_requests = df_servers['total_requests'].sum()
            st.metric("Total Requests", total_requests)
        
        with col2:
            active_requests = df_servers['active_requests'].sum()
            st.metric("Active Requests", active_requests)
        
        with col3:
            avg_success_rate = df_servers['success_rate'].mean()
            st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
        
        with col4:
            avg_latency = df_servers['avg_latency'].mean()
            st.metric("Avg Latency", f"{avg_latency:.3f}s")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Request distribution
            fig_requests = px.pie(df_servers, values="total_requests", names="server",
                                title="Request Distribution by Server")
            st.plotly_chart(fig_requests, use_container_width=True)
        
        with col2:
            # Latency comparison
            fig_latency = px.bar(df_servers, x="server", y="avg_latency",
                               title="Average Latency by Server (seconds)",
                               color="avg_latency", color_continuous_scale="RdYlBu")
            st.plotly_chart(fig_latency, use_container_width=True)
    
    # GPU Performance Analysis
    if perf_data:
        st.markdown("### 🎮 GPU Performance Analysis")
        
        df_gpu_perf = pd.DataFrame(perf_data)
        
        # GPU efficiency analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # GPU Utilization vs Active Requests
            fig_efficiency = px.scatter(df_gpu_perf, x="active_requests", y="utilization",
                                      color="server", size="memory_used",
                                      title="GPU Utilization vs Active Requests",
                                      hover_data=["gpu"])
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        with col2:
            # Memory usage efficiency
            df_gpu_perf['memory_usage_percent'] = (df_gpu_perf['memory_used'] / df_gpu_perf['memory_total']) * 100
            fig_memory_eff = px.bar(df_gpu_perf, x="server", y="memory_usage_percent",
                                  title="GPU Memory Usage Efficiency (%)",
                                  color="memory_usage_percent", 
                                  color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_memory_eff, use_container_width=True)

def show_realtime_gpu_monitoring():
    """Real-time GPU monitoring section"""
    st.markdown("### 🔄 Real-Time GPU Monitoring")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=False)
    
    with col2:
        refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1)
    
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Real-time metrics
    if auto_refresh:
        # Auto-refresh placeholder (would need JavaScript or periodic rerun)
        st.info(f"Auto-refreshing every {refresh_interval} seconds")
        # Note: Streamlit doesn't support true auto-refresh without custom components
        # This would need additional implementation
    
    # Current GPU status
    detailed_status = get_detailed_server_status()
    
    if detailed_status:
        # Create real-time GPU metrics
        gpu_metrics = []
        
        for server_name, stats in detailed_status.items():
            if 'error' in stats:
                continue
                
            gpu_info = stats.get('gpu_info', {})
            for gpu_name, gpu_data in gpu_info.items():
                gpu_metrics.append({
                    'Server': server_name,
                    'GPU': gpu_name,
                    'Utilization (%)': f"{gpu_data.get('utilization', 0):.1f}",
                    'Memory Used (GB)': f"{gpu_data.get('memory_cached', 0) / (1024**3):.1f}",
                    'Memory Total (GB)': f"{gpu_data.get('total_memory', 0) / (1024**3):.1f}",
                    'Active Requests': stats.get('active_requests', 0),
                    'Status': '🟢' if stats.get('status') == 'healthy' else '🔴'
                })
        
        if gpu_metrics:
            # Display as table
            df_metrics = pd.DataFrame(gpu_metrics)
            st.dataframe(df_metrics, use_container_width=True)
            
            # Quick stats
            total_gpus = len(gpu_metrics)
            healthy_gpus = sum(1 for m in gpu_metrics if m['Status'] == '🟢')
            avg_utilization = sum(float(m['Utilization (%)']) for m in gpu_metrics) / total_gpus
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total GPUs", total_gpus)
            with col2:
                st.metric("Healthy GPUs", healthy_gpus)
            with col3:
                st.metric("Avg GPU Utilization", f"{avg_utilization:.1f}%")

def show_gpu_allocation_optimizer():
    """GPU allocation optimizer recommendations"""
    st.markdown("### 🎯 GPU Allocation Optimizer")
    
    detailed_status = get_detailed_server_status()
    
    if not detailed_status:
        st.warning("No server data available for optimization")
        return
    
    # Analyze current allocation
    recommendations = []
    gpu_analysis = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
            
        gpu_info = stats.get('gpu_info', {})
        active_requests = stats.get('active_requests', 0)
        
        for gpu_name, gpu_data in gpu_info.items():
            utilization = gpu_data.get('utilization', 0)
            memory_total = gpu_data.get('total_memory', 0)
            memory_used = gpu_data.get('memory_cached', 0)
            memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
            
            gpu_analysis.append({
                'server': server_name,
                'gpu': gpu_name,
                'utilization': utilization,
                'memory_percent': memory_percent,
                'active_requests': active_requests,
                'efficiency': utilization / max(active_requests, 1)  # requests per utilization
            })
            
            # Generate recommendations
            if utilization > 90:
                recommendations.append({
                    'type': 'warning',
                    'server': server_name,
                    'gpu': gpu_name,
                    'message': f"High GPU utilization ({utilization:.1f}%) - consider load balancing"
                })
            elif utilization < 10 and active_requests > 0:
                recommendations.append({
                    'type': 'info',
                    'server': server_name,
                    'gpu': gpu_name,
                    'message': f"Low GPU utilization ({utilization:.1f}%) with active requests - check efficiency"
                })
            elif memory_percent > 95:
                recommendations.append({
                    'type': 'error',
                    'server': server_name,
                    'gpu': gpu_name,
                    'message': f"Critical memory usage ({memory_percent:.1f}%) - immediate attention needed"
                })
            elif memory_percent > 80:
                recommendations.append({
                    'type': 'warning',
                    'server': server_name,
                    'gpu': gpu_name,
                    'message': f"High memory usage ({memory_percent:.1f}%) - monitor closely"
                })
    
    # Display recommendations
    if recommendations:
        st.markdown("#### 💡 Optimization Recommendations")
        
        for rec in recommendations:
            if rec['type'] == 'error':
                st.error(f"🚨 {rec['server']} ({rec['gpu']}): {rec['message']}")
            elif rec['type'] == 'warning':
                st.warning(f"⚠️ {rec['server']} ({rec['gpu']}): {rec['message']}")
            else:
                st.info(f"ℹ️ {rec['server']} ({rec['gpu']}): {rec['message']}")
    else:
        st.success("✅ All GPUs are operating within optimal parameters")
    
    # Efficiency analysis
    if gpu_analysis:
        st.markdown("#### 📈 Efficiency Analysis")
        
        df_analysis = pd.DataFrame(gpu_analysis)
        
        # Find most and least efficient GPUs
        if len(df_analysis) > 1:
            most_efficient = df_analysis.loc[df_analysis['efficiency'].idxmax()]
            least_efficient = df_analysis.loc[df_analysis['efficiency'].idxmin()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"🏆 Most Efficient: {most_efficient['server']} ({most_efficient['gpu']})")
                st.metric("Efficiency Score", f"{most_efficient['efficiency']:.2f}")
            
            with col2:
                st.error(f"📉 Needs Attention: {least_efficient['server']} ({least_efficient['gpu']})")
                st.metric("Efficiency Score", f"{least_efficient['efficiency']:.2f}")
        
        # Efficiency chart
        fig_efficiency = px.scatter(df_analysis, x="memory_percent", y="utilization",
                                  color="efficiency", size="active_requests",
                                  title="GPU Efficiency Matrix",
                                  labels={
                                      "memory_percent": "Memory Usage (%)",
                                      "utilization": "GPU Utilization (%)",
                                      "efficiency": "Efficiency Score"
                                  },
                                  hover_data=["server", "gpu"])
        
        # Add quadrant lines
        fig_efficiency.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig_efficiency.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_efficiency, use_container_width=True)

def show_gpu_load_balancing_insights():
    """Show insights for GPU-based load balancing"""
    st.markdown("### ⚖️ Load Balancing Insights")
    
    detailed_status = get_detailed_server_status()
    
    if not detailed_status:
        st.warning("No data available for load balancing analysis")
        return
    
    # Analyze load distribution
    load_data = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
            
        gpu_info = stats.get('gpu_info', {})
        active_requests = stats.get('active_requests', 0)
        total_requests = stats.get('total_requests', 0)
        success_rate = stats.get('success_rate', 0)
        avg_latency = stats.get('average_latency', 0)
        
        # Calculate average GPU utilization for this server
        gpu_utilizations = [gpu_data.get('utilization', 0) for gpu_data in gpu_info.values()]
        avg_gpu_util = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0
        
        # Calculate average memory usage
        memory_usages = []
        for gpu_data in gpu_info.values():
            total_mem = gpu_data.get('total_memory', 0)
            used_mem = gpu_data.get('memory_cached', 0)
            if total_mem > 0:
                memory_usages.append((used_mem / total_mem) * 100)
        
        avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        
        load_data.append({
            'server': server_name,
            'active_requests': active_requests,
            'total_requests': total_requests,
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'avg_gpu_utilization': avg_gpu_util,
            'avg_memory_usage': avg_memory_usage,
            'gpu_count': len(gpu_info),
            'load_score': (active_requests * 0.3) + (avg_gpu_util * 0.4) + (avg_memory_usage * 0.3)
        })
    
    if not load_data:
        st.warning("No valid server data for analysis")
        return
    
    df_load = pd.DataFrame(load_data)
    
    # Current load balancing strategy
    if st.session_state.coordinator:
        try:
            health = st.session_state.coordinator.get_health_status()
            current_strategy = health.get('load_balancing_strategy', 'unknown')
            st.info(f"Current Strategy: **{current_strategy.replace('_', ' ').title()}**")
        except:
            st.info("Current Strategy: **Unknown**")
    
    # Load distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Request distribution
        fig_load = px.bar(df_load, x="server", y="active_requests",
                         title="Current Active Requests Distribution",
                         color="active_requests", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig_load, use_container_width=True)
    
    with col2:
        # GPU utilization distribution
        fig_gpu = px.bar(df_load, x="server", y="avg_gpu_utilization",
                        title="Average GPU Utilization Distribution",
                        color="avg_gpu_utilization", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig_gpu, use_container_width=True)
    
    # Load balancing recommendations
    st.markdown("#### 💡 Load Balancing Recommendations")
    
    # Calculate load imbalance
    max_load = df_load['load_score'].max()
    min_load = df_load['load_score'].min()
    load_imbalance = max_load - min_load
    
    if load_imbalance > 30:  # Threshold for significant imbalance
        overloaded_server = df_load.loc[df_load['load_score'].idxmax(), 'server']
        underloaded_server = df_load.loc[df_load['load_score'].idxmin(), 'server']
        
        st.warning(f"⚠️ Significant load imbalance detected!")
        st.write(f"- **Overloaded**: {overloaded_server} (Load Score: {max_load:.1f})")
        st.write(f"- **Underutilized**: {underloaded_server} (Load Score: {min_load:.1f})")
        st.write("**Recommendation**: Consider routing new requests to underutilized servers")
    else:
        st.success("✅ Load is well balanced across servers")
    
    # Strategy recommendations based on current state
    st.markdown("#### 🎯 Strategy Recommendations")
    
    # Analyze current conditions
    avg_gpu_util = df_load['avg_gpu_utilization'].mean()
    avg_memory_usage = df_load['avg_memory_usage'].mean()
    total_active_requests = df_load['active_requests'].sum()
    latency_variance = df_load['avg_latency'].std()
    
    strategy_recommendations = []
    
    if avg_gpu_util > 80:
        strategy_recommendations.append({
            'strategy': 'LEAST_GPU_UTILIZATION',
            'reason': f'High average GPU utilization ({avg_gpu_util:.1f}%)',
            'priority': 'High'
        })
    
    if avg_memory_usage > 75:
        strategy_recommendations.append({
            'strategy': 'LEAST_MEMORY_USAGE',
            'reason': f'High average memory usage ({avg_memory_usage:.1f}%)',
            'priority': 'High'
        })
    
    if latency_variance > 0.1:  # High latency variance
        strategy_recommendations.append({
            'strategy': 'LEAST_LATENCY',
            'reason': f'High latency variance ({latency_variance:.3f}s)',
            'priority': 'Medium'
        })
    
    if total_active_requests > 50:
        strategy_recommendations.append({
            'strategy': 'LEAST_BUSY',
            'reason': f'High total active requests ({total_active_requests})',
            'priority': 'Medium'
        })
    
    if not strategy_recommendations:
        strategy_recommendations.append({
            'strategy': 'ROUND_ROBIN',
            'reason': 'System is well balanced, simple round-robin is sufficient',
            'priority': 'Low'
        })
    
    for rec in strategy_recommendations:
        if rec['priority'] == 'High':
            st.error(f"🔴 **{rec['strategy']}**: {rec['reason']}")
        elif rec['priority'] == 'Medium':
            st.warning(f"🟡 **{rec['strategy']}**: {rec['reason']}")
        else:
            st.info(f"🟢 **{rec['strategy']}**: {rec['reason']}")
    
    # Load balancing effectiveness metrics
    st.markdown("#### 📊 Load Balancing Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Load Imbalance Score", f"{load_imbalance:.1f}")
    
    with col2:
        st.metric("Avg GPU Utilization", f"{avg_gpu_util:.1f}%")
    
    with col3:
        st.metric("Avg Memory Usage", f"{avg_memory_usage:.1f}%")
    
    with col4:
        efficiency_score = 100 - (load_imbalance / 2)  # Simple efficiency calculation
        st.metric("Balancing Efficiency", f"{efficiency_score:.1f}%")

def update_server_status_display():
    """Updated server status display function that replaces the original"""
    st.markdown("### 🌐 Enhanced Server Status & GPU Monitoring")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖥️ Server Overview", 
        "🎮 GPU Monitoring", 
        "📊 Performance", 
        "⚖️ Load Balancing"
    ])
    
    with tab1:
        show_enhanced_server_status()
    
    with tab2:
        show_realtime_gpu_monitoring()
        st.markdown("---")
        show_gpu_allocation_optimizer()
    
    with tab3:
        detailed_status = get_detailed_server_status()
        show_performance_tab(detailed_status)
    
    with tab4:
        show_gpu_load_balancing_insights()

# Add GPU-specific health check function
def run_gpu_health_check():
    """Run comprehensive GPU health check on all servers"""
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Running comprehensive GPU health check..."):
        try:
            detailed_status = get_detailed_server_status()
            
            if not detailed_status:
                st.error("❌ No server status available")
                return
            
            # Analyze GPU health
            gpu_health_results = []
            overall_health = True
            
            for server_name, stats in detailed_status.items():
                if 'error' in stats:
                    gpu_health_results.append({
                        'server': server_name,
                        'status': 'Error',
                        'message': stats['error'],
                        'healthy': False
                    })
                    overall_health = False
                    continue
                
                gpu_info = stats.get('gpu_info', {})
                
                if not gpu_info:
                    gpu_health_results.append({
                        'server': server_name,
                        'status': 'No GPU Info',
                        'message': 'GPU information not available',
                        'healthy': False
                    })
                    overall_health = False
                    continue
                
                # Check each GPU
                server_gpu_healthy = True
                gpu_messages = []
                
                for gpu_name, gpu_data in gpu_info.items():
                    utilization = gpu_data.get('utilization', 0)
                    memory_total = gpu_data.get('total_memory', 0)
                    memory_used = gpu_data.get('memory_cached', 0)
                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                    
                    # Health checks
                    if memory_percent > 95:
                        gpu_messages.append(f"{gpu_name}: Critical memory usage ({memory_percent:.1f}%)")
                        server_gpu_healthy = False
                    elif memory_percent > 85:
                        gpu_messages.append(f"{gpu_name}: High memory usage ({memory_percent:.1f}%)")
                    
                    if utilization > 95:
                        gpu_messages.append(f"{gpu_name}: Very high utilization ({utilization:.1f}%)")
                    
                    if memory_total == 0:
                        gpu_messages.append(f"{gpu_name}: Memory information unavailable")
                        server_gpu_healthy = False
                
                if server_gpu_healthy and not gpu_messages:
                    gpu_messages.append("All GPUs operating normally")
                
                gpu_health_results.append({
                    'server': server_name,
                    'status': 'Healthy' if server_gpu_healthy else 'Issues',
                    'message': '; '.join(gpu_messages),
                    'healthy': server_gpu_healthy
                })
                
                if not server_gpu_healthy:
                    overall_health = False
            
            # Display results
            if overall_health:
                st.success("✅ All GPU systems are healthy")
            else:
                st.error("❌ GPU health issues detected")
            
            # Detailed results
            for result in gpu_health_results:
                if result['healthy']:
                    st.success(f"🟢 {result['server']}: {result['message']}")
                else:
                    st.error(f"🔴 {result['server']}: {result['message']}")
            
            # Summary metrics
            healthy_servers = sum(1 for r in gpu_health_results if r['healthy'])
            total_servers = len(gpu_health_results)
            
            st.info(f"Health Summary: {healthy_servers}/{total_servers} servers healthy")
            
        except Exception as e:
            st.error(f"Health check failed: {str(e)}")
            st.exception(e)
    # Enhanced GPU Status Functions for Streamlit - Part 3
# Integration updates and enhanced dashboard functions

def show_enhanced_system_health():
    """Enhanced system health page with comprehensive GPU monitoring"""
    st.markdown('<div class="sub-header">⚙️ Enhanced API System Health & GPU Monitoring</div>', unsafe_allow_html=True)
    
    if not COORDINATOR_AVAILABLE:
        st.error("🔴 System Status: API Coordinator Not Available")
        st.markdown("### Import Error")
        st.code(st.session_state.get('import_error', 'Unknown import error'))
        st.info("Please ensure the api_opulence_coordinator module is properly installed and configured.")
        return
    
    # Server configuration section
    configure_model_servers()
    
    st.markdown("---")
    
    if not st.session_state.coordinator:
        st.warning("🟡 System Status: Not Initialized")
        if st.button("🔄 Initialize API System"):
            with st.spinner("Initializing API system..."):
                try:
                    success = safe_run_async(init_api_coordinator())
                    if success and not isinstance(success, dict):
                        st.success("✅ API system initialized successfully")
                        st.rerun()
                    else:
                        error_msg = success.get('error') if isinstance(success, dict) else st.session_state.initialization_status
                        st.error(f"❌ Initialization failed: {error_msg}")
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
        return
    
    # Enhanced server status display
    update_server_status_display()
    
    st.markdown("---")
    
    # System controls with GPU-specific actions
    st.markdown("### 🛠️ System Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh All"):
            st.rerun()
    
    with col2:
        if st.button("🎮 GPU Health Check"):
            run_gpu_health_check()
    
    with col3:
        if st.button("🧹 Clean Memory"):
            try:
                # Clear GPU memory if possible
                detailed_status = get_detailed_server_status()
                cleaned_servers = 0
                
                for server_name, stats in detailed_status.items():
                    if 'error' not in stats and stats.get('endpoint'):
                        try:
                            # Call a hypothetical cleanup endpoint
                            endpoint = stats['endpoint']
                            response = requests.post(f"{endpoint}/cleanup", timeout=10)
                            if response.status_code == 200:
                                cleaned_servers += 1
                        except:
                            pass  # Server may not support cleanup endpoint
                
                st.success(f"Memory cleanup attempted on {cleaned_servers} servers")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clean memory: {str(e)}")
    
    with col4:
        if st.button("📊 Export GPU Report"):
            export_gpu_report()

def export_gpu_report():
    """Export comprehensive GPU status report"""
    try:
        detailed_status = get_detailed_server_status()
        
        if not detailed_status:
            st.warning("No data available for export")
            return
        
        # Generate comprehensive report
        report_data = {
            "report_metadata": {
                "timestamp": dt.now().isoformat(),
                "report_type": "gpu_status_report",
                "coordinator_type": "api_based",
                "total_servers": len(detailed_status)
            },
            "server_summary": [],
            "gpu_details": [],
            "performance_metrics": [],
            "health_analysis": []
        }
        
        # Collect data for each server
        for server_name, stats in detailed_status.items():
            server_summary = {
                "server_name": server_name,
                "status": stats.get('status', 'unknown'),
                "endpoint": stats.get('endpoint', 'unknown'),
                "active_requests": stats.get('active_requests', 0),
                "total_requests": stats.get('total_requests', 0),
                "success_rate": stats.get('success_rate', 0),
                "average_latency": stats.get('average_latency', 0),
                "uptime": stats.get('uptime', 0),
                "has_error": 'error' in stats,
                "error_message": stats.get('error', '')
            }
            report_data["server_summary"].append(server_summary)
            
            if 'error' not in stats:
                gpu_info = stats.get('gpu_info', {})
                for gpu_name, gpu_data in gpu_info.items():
                    gpu_detail = {
                        "server_name": server_name,
                        "gpu_name": gpu_name,
                        "gpu_model": gpu_data.get('name', 'Unknown'),
                        "compute_capability": gpu_data.get('compute_capability', 'Unknown'),
                        "total_memory_gb": gpu_data.get('total_memory', 0) / (1024**3),
                        "allocated_memory_gb": gpu_data.get('memory_allocated', 0) / (1024**3),
                        "cached_memory_gb": gpu_data.get('memory_cached', 0) / (1024**3),
                        "free_memory_gb": gpu_data.get('memory_free', 0) / (1024**3),
                        "utilization_percent": gpu_data.get('utilization', 0),
                        "memory_usage_percent": (gpu_data.get('memory_cached', 0) / gpu_data.get('total_memory', 1)) * 100
                    }
                    report_data["gpu_details"].append(gpu_detail)
                
                # Performance metrics
                perf_metric = {
                    "server_name": server_name,
                    "requests_per_second": stats.get('total_requests', 0) / max(stats.get('uptime', 1), 1),
                    "average_gpu_utilization": sum(gpu_data.get('utilization', 0) for gpu_data in gpu_info.values()) / max(len(gpu_info), 1),
                    "average_memory_usage": sum((gpu_data.get('memory_cached', 0) / gpu_data.get('total_memory', 1)) * 100 for gpu_data in gpu_info.values()) / max(len(gpu_info), 1),
                    "gpu_count": len(gpu_info)
                }
                report_data["performance_metrics"].append(perf_metric)
        
        # Generate health analysis
        healthy_servers = sum(1 for s in report_data["server_summary"] if s["status"] == "healthy" and not s["has_error"])
        total_servers = len(report_data["server_summary"])
        
        health_analysis = {
            "overall_health": "healthy" if healthy_servers == total_servers else "degraded",
            "healthy_servers": healthy_servers,
            "total_servers": total_servers,
            "health_percentage": (healthy_servers / total_servers * 100) if total_servers > 0 else 0,
            "total_gpus": len(report_data["gpu_details"]),
            "average_gpu_utilization": sum(gpu["utilization_percent"] for gpu in report_data["gpu_details"]) / max(len(report_data["gpu_details"]), 1),
            "average_memory_usage": sum(gpu["memory_usage_percent"] for gpu in report_data["gpu_details"]) / max(len(report_data["gpu_details"]), 1)
        }
        report_data["health_analysis"] = health_analysis
        
        # Create downloadable report
        report_json = json.dumps(report_data, indent=2)
        
        st.download_button(
            "📥 Download GPU Status Report",
            report_json,
            file_name=f"gpu_status_report_{dt.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Also create a human-readable summary
        summary_lines = [
            "# GPU Status Report Summary",
            f"Generated: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Overview",
            f"- Total Servers: {total_servers}",
            f"- Healthy Servers: {healthy_servers}",
            f"- System Health: {health_analysis['overall_health'].upper()}",
            f"- Total GPUs: {health_analysis['total_gpus']}",
            f"- Average GPU Utilization: {health_analysis['average_gpu_utilization']:.1f}%",
            f"- Average Memory Usage: {health_analysis['average_memory_usage']:.1f}%",
            "",
            "## Server Details"
        ]
        
        for server in report_data["server_summary"]:
            summary_lines.extend([
                f"### {server['server_name']}",
                f"- Status: {server['status']}",
                f"- Active Requests: {server['active_requests']}",
                f"- Success Rate: {server['success_rate']:.1f}%",
                f"- Uptime: {server['uptime']:.0f}s",
                ""
            ])
        
        summary_text = "\n".join(summary_lines)
        
        st.download_button(
            "📄 Download Summary Report",
            summary_text,
            file_name=f"gpu_summary_{dt.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        
        st.success("GPU reports generated successfully!")
        
    except Exception as e:
        st.error(f"Failed to generate GPU report: {str(e)}")

def show_enhanced_dashboard():
    """Enhanced dashboard with GPU monitoring integration"""
    st.markdown('<div class="sub-header">🌐 Enhanced API System Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.warning("System not initialized")
        return
    
    # Get comprehensive system status
    try:
        health = st.session_state.coordinator.get_health_status()
        stats_result = safe_run_async(st.session_state.coordinator.get_statistics())
        detailed_status = get_detailed_server_status()
        
        # Enhanced system health indicator
        available_servers = health.get('available_servers', 0)
        total_servers = health.get('total_servers', 0)
        
        if health.get("status") == "healthy" and available_servers == total_servers:
            st.success(f"🟢 All Systems Operational - {available_servers}/{total_servers} servers healthy")
        elif available_servers > 0:
            st.warning(f"⚠️ Partial Service - {available_servers}/{total_servers} servers healthy")
        else:
            st.error(f"🔴 Service Degraded - {available_servers}/{total_servers} servers healthy")
        
        # GPU System Overview
        gpu_overview = []
        total_gpus = 0
        healthy_gpus = 0
        total_gpu_utilization = 0
        total_memory_usage = 0
        
        for server_name, stats in detailed_status.items():
            if 'error' not in stats:
                gpu_info = stats.get('gpu_info', {})
                total_gpus += len(gpu_info)
                
                for gpu_name, gpu_data in gpu_info.items():
                    if stats.get('status') == 'healthy':
                        healthy_gpus += 1
                    total_gpu_utilization += gpu_data.get('utilization', 0)
                    
                    memory_total = gpu_data.get('total_memory', 0)
                    memory_used = gpu_data.get('memory_cached', 0)
                    if memory_total > 0:
                        total_memory_usage += (memory_used / memory_total) * 100
        
        avg_gpu_utilization = total_gpu_utilization / max(total_gpus, 1)
        avg_memory_usage = total_memory_usage / max(total_gpus, 1)
        
        # Enhanced metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Active Agents", health.get("active_agents", 0))
        
        with col2:
            st.metric("Available Servers", f"{available_servers}/{total_servers}")
        
        with col3:
            st.metric("Healthy GPUs", f"{healthy_gpus}/{total_gpus}")
        
        with col4:
            st.metric("Avg GPU Utilization", f"{avg_gpu_utilization:.1f}%")
        
        with col5:
            st.metric("Avg GPU Memory", f"{avg_memory_usage:.1f}%")
        
        # Enhanced server status with tabs
        st.markdown("### 🖥️ Server & GPU Status")
        update_server_status_display()
        
        # System activity and performance trends
        st.markdown("### 📈 System Activity")
        
        if st.session_state.processing_history:
            df_activity = pd.DataFrame(st.session_state.processing_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Processing activity over time
                fig_activity = px.line(df_activity, x="timestamp", y="files_count",
                                     title="File Processing Activity",
                                     markers=True)
                st.plotly_chart(fig_activity, use_container_width=True)
            
            with col2:
                # Processing time trends
                if 'processing_time' in df_activity.columns:
                    fig_time = px.bar(df_activity, x="timestamp", y="processing_time",
                                    title="Processing Time Trends",
                                    color="status")
                    st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No processing history available")
        
        # Quick actions
        st.markdown("### 🎛️ Quick Actions")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        with col2:
            if st.button("🎮 GPU Check"):
                run_gpu_health_check()
        
        with col3:
            if st.button("📊 Full Status"):
                # Navigate to system health page
                st.info("Navigate to System Health tab for detailed status")
        
        with col4:
            if st.button("⚖️ Load Balance"):
                # Show load balancing recommendations
                with st.expander("Load Balancing Analysis", expanded=True):
                    show_gpu_load_balancing_insights()
        
        with col5:
            if st.button("📈 Export Report"):
                export_gpu_report()
        
    except Exception as e:
        st.error(f"Error loading enhanced dashboard: {str(e)}")
        st.exception(e)

# Updated main application function with enhanced GPU monitoring
def main_enhanced():
    """Enhanced main application function with comprehensive GPU monitoring"""
    
    # Header
    st.markdown('<div class="main-header">🌐 Opulence Enhanced API-Based Deep Research Agent</div>', unsafe_allow_html=True)
    
    # Show import status
    if not COORDINATOR_AVAILABLE:
        st.error("⚠️ API Coordinator module not available - Running in demo mode")
        if st.button("🐛 Toggle Debug Mode"):
            st.session_state.debug_mode = not st.session_state.debug_mode
            st.rerun()
    
    # Sidebar navigation with enhanced options
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/059669/ffffff?text=OPULENCE", use_container_width=True)
        
        page = st.selectbox(
            "Navigation",
            ["🏠 Enhanced Dashboard", "📂 File Upload", "💬 Enhanced Chat", "🔍 Enhanced Analysis", 
             "⚙️ System Health", "🎮 GPU Monitoring", "🤖 Agent Status", "📊 Full Monitoring"] 
        )
        
        # Quick actions with GPU focus
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh System"):
            st.rerun()
        
        if st.button("🎮 Quick GPU Check"):
            if st.session_state.coordinator:
                run_gpu_health_check()
            else:
                st.error("System not initialized")
        
        # Enhanced system status indicator
        if COORDINATOR_AVAILABLE and st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                available_servers = health.get('available_servers', 0)
                total_servers = health.get('total_servers', 0)
                
                # Get GPU status
                detailed_status = get_detailed_server_status()
                total_gpus = sum(len(stats.get('gpu_info', {})) for stats in detailed_status.values() if 'error' not in stats)
                healthy_gpus = sum(
                    len(stats.get('gpu_info', {})) 
                    for stats in detailed_status.values() 
                    if 'error' not in stats and stats.get('status') == 'healthy'
                )
                
                if available_servers > 0 and healthy_gpus > 0:
                    st.success(f"🟢 System Healthy")
                    st.caption(f"Servers: {available_servers}/{total_servers}")
                    st.caption(f"GPUs: {healthy_gpus}/{total_gpus}")
                elif available_servers > 0:
                    st.warning(f"🟡 Partial Service")
                    st.caption(f"Servers: {available_servers}/{total_servers}")
                    st.caption(f"GPUs: {healthy_gpus}/{total_gpus}")
                else:
                    st.error("🔴 Service Down")
                    st.caption("No servers available")
            except:
                st.success("🟢 API System Ready")
        elif COORDINATOR_AVAILABLE:
            st.warning("🟡 API System Not Initialized")
        else:
            st.error("🔴 Demo Mode")
        
        # GPU utilization quick view
        if st.session_state.coordinator:
            try:
                st.markdown("### 🎮 GPU Quick View")
                detailed_status = get_detailed_server_status()
                
                gpu_utils = []
                for server_name, stats in detailed_status.items():
                    if 'error' not in stats:
                        gpu_info = stats.get('gpu_info', {})
                        for gpu_name, gpu_data in gpu_info.items():
                            utilization = gpu_data.get('utilization', 0)
                            gpu_utils.append(utilization)
                            
                            # Show quick GPU status
                            if utilization > 90:
                                st.error(f"🔴 {server_name}: {utilization:.0f}%")
                            elif utilization > 70:
                                st.warning(f"🟡 {server_name}: {utilization:.0f}%")
                            else:
                                st.success(f"🟢 {server_name}: {utilization:.0f}%")
                
                if gpu_utils:
                    avg_util = sum(gpu_utils) / len(gpu_utils)
                    st.metric("Avg GPU Util", f"{avg_util:.1f}%")
                    
            except Exception as e:
                st.caption(f"GPU status unavailable")
        
        # Show example queries
        show_example_queries()
        
        # Show debug info if enabled
        show_debug_info()
    
    # Main content based on selected page
    try:
        if page == "🏠 Enhanced Dashboard":
            show_enhanced_dashboard()
        elif page == "📂 File Upload":
            show_file_upload()
        elif page == "💬 Enhanced Chat":
            show_enhanced_chat_analysis()
        elif page == "🔍 Enhanced Analysis":
            show_enhanced_component_analysis()
        elif page == "⚙️ System Health":
            show_enhanced_system_health()
        elif page == "🤖 Agent Status":
            show_agent_status_page()  # New
        elif page == "🎮 GPU Monitoring":
            show_dedicated_gpu_monitoring_page()
        elif page == "📊 Full Monitoring":
            show_complete_monitoring_dashboard()
    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        st.exception(e)
    
    # Enhanced footer with GPU info
    show_enhanced_footer()

def show_sidebar_quick_status():
    """Quick status display in sidebar"""
    if COORDINATOR_AVAILABLE and st.session_state.coordinator:
        try:
            # System health
            health = st.session_state.coordinator.get_health_status()
            available_servers = health.get('available_servers', 0)
            total_servers = health.get('total_servers', 0)
            
            if available_servers > 0:
                st.success(f"🟢 System: {available_servers}/{total_servers}")
            else:
                st.error("🔴 System: Down")
            
            # GPU quick status
            try:
                detailed_status = get_detailed_server_status()
                total_gpus = sum(len(stats.get('gpu_info', {})) for stats in detailed_status.values() if 'error' not in stats)
                if total_gpus > 0:
                    gpu_utils = []
                    for stats in detailed_status.values():
                        if 'error' not in stats:
                            gpu_info = stats.get('gpu_info', {})
                            for gpu_data in gpu_info.values():
                                gpu_utils.append(gpu_data.get('utilization', 0))
                    
                    if gpu_utils:
                        avg_util = sum(gpu_utils) / len(gpu_utils)
                        st.info(f"🎮 GPUs: {avg_util:.0f}% avg")
            except:
                pass
            
            # Agent quick status
            try:
                agent_status = get_agent_status_info()
                if 'error' not in agent_status:
                    active_agents = agent_status.get('coordinator_info', {}).get('active_agents', 0)
                    st.info(f"🤖 Agents: {active_agents} active")
            except:
                pass
                
        except:
            st.warning("🟡 Status: Unknown")
    elif COORDINATOR_AVAILABLE:
        st.warning("🟡 Not Initialized")
    else:
        st.error("🔴 Demo Mode")
def get_agent_status_info():
    """Get comprehensive agent status information"""
    if not st.session_state.coordinator:
        return {}
    
    try:
        agent_status = {
            'coordinator_info': {
                'type': 'api_based',
                'initialization_time': getattr(st.session_state.coordinator, 'start_time', time.time()),
                'active_agents': len(st.session_state.coordinator.agents),
                'available_agent_types': [
                    'code_parser', 'vector_index', 'data_loader', 
                    'lineage_analyzer', 'logic_analyzer', 'documentation',
                    'db2_comparator', 'chat_agent'
                ]
            },
            'agents': {},
            'agent_health': {},
            'agent_performance': {},
            'database_status': {}
        }
        
        # Get information about each active agent
        for agent_type, agent_instance in st.session_state.coordinator.agents.items():
            try:
                agent_info = {
                    'type': agent_type,
                    'status': 'active',
                    'created_time': getattr(agent_instance, '_created_time', 'unknown'),
                    'last_used': getattr(agent_instance, '_last_used', 'unknown'),
                    'gpu_id': getattr(agent_instance, 'gpu_id', 'api_based'),
                    'db_path': getattr(agent_instance, 'db_path', 'unknown'),
                    'coordinator_ref': hasattr(agent_instance, 'coordinator'),
                    'engine_context': 'api_based',
                    'methods_available': [method for method in dir(agent_instance) 
                                        if not method.startswith('_') and callable(getattr(agent_instance, method))]
                }
                
                # Try to get agent-specific metrics
                if hasattr(agent_instance, 'get_stats'):
                    try:
                        agent_info['stats'] = agent_instance.get_stats()
                    except:
                        agent_info['stats'] = 'unavailable'
                
                agent_status['agents'][agent_type] = agent_info
                
            except Exception as e:
                agent_status['agents'][agent_type] = {
                    'type': agent_type,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Check database connectivity for agents
        try:
            db_path = st.session_state.coordinator.db_path
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check key tables for agent operations
                tables_status = {}
                key_tables = ['program_chunks', 'file_metadata', 'field_lineage', 'vector_embeddings']
                
                for table in key_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        tables_status[table] = {
                            'status': 'available',
                            'record_count': count
                        }
                    except sqlite3.OperationalError:
                        tables_status[table] = {
                            'status': 'missing',
                            'record_count': 0
                        }
                
                conn.close()
                agent_status['database_status'] = {
                    'status': 'connected',
                    'path': db_path,
                    'size_mb': os.path.getsize(db_path) / (1024*1024),
                    'tables': tables_status
                }
            else:
                agent_status['database_status'] = {
                    'status': 'missing',
                    'path': db_path
                }
        except Exception as e:
            agent_status['database_status'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return agent_status
        
    except Exception as e:
        return {
            'error': f"Failed to get agent status: {str(e)}",
            'coordinator_available': st.session_state.coordinator is not None
        }

def show_agent_status_overview():
    """Show agent status overview section"""
    st.markdown("### 🤖 Agent Status Overview")
    
    agent_status = get_agent_status_info()
    
    if 'error' in agent_status:
        st.error(f"❌ Agent Status Error: {agent_status['error']}")
        return
    
    coordinator_info = agent_status.get('coordinator_info', {})
    
    # Agent overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        active_agents = coordinator_info.get('active_agents', 0)
        st.metric("Active Agents", active_agents)
    
    with col2:
        available_types = len(coordinator_info.get('available_agent_types', []))
        st.metric("Available Types", available_types)
    
    with col3:
        coordinator_type = coordinator_info.get('type', 'unknown')
        st.metric("Coordinator Type", coordinator_type.replace('_', ' ').title())
    
    with col4:
        init_time = coordinator_info.get('initialization_time', time.time())
        uptime = time.time() - init_time
        st.metric("Uptime", f"{uptime:.0f}s")
    
    # Agent health status
    agents = agent_status.get('agents', {})
    if agents:
        st.markdown("#### 🔍 Agent Health Status")
        
        healthy_agents = sum(1 for agent in agents.values() if agent.get('status') == 'active')
        total_agents = len(agents)
        
        if healthy_agents == total_agents:
            st.success(f"✅ All {total_agents} agents are healthy")
        else:
            st.warning(f"⚠️ {healthy_agents}/{total_agents} agents are healthy")
        
        # Show each agent status
        cols = st.columns(min(len(agents), 4))
        for i, (agent_type, agent_info) in enumerate(agents.items()):
            with cols[i % 4]:
                status = agent_info.get('status', 'unknown')
                if status == 'active':
                    st.success(f"🟢 {agent_type.replace('_', ' ').title()}")
                else:
                    st.error(f"🔴 {agent_type.replace('_', ' ').title()}")
                    if 'error' in agent_info:
                        st.caption(f"Error: {agent_info['error']}")
    else:
        st.info("No agents currently active")

def show_detailed_agent_status():
    """Show detailed agent status with full information"""
    st.markdown("### 🔧 Detailed Agent Status")
    
    agent_status = get_agent_status_info()
    
    if 'error' in agent_status:
        st.error(f"Cannot retrieve agent details: {agent_status['error']}")
        return
    
    agents = agent_status.get('agents', {})
    
    if not agents:
        st.info("No agents currently instantiated")
        return
    
    # Create tabs for each agent
    agent_names = list(agents.keys())
    if len(agent_names) == 1:
        # Single agent, no tabs needed
        show_single_agent_details(agent_names[0], agents[agent_names[0]])
    else:
        # Multiple agents, use tabs
        tabs = st.tabs([name.replace('_', ' ').title() for name in agent_names])
        
        for i, (agent_name, agent_info) in enumerate(agents.items()):
            with tabs[i]:
                show_single_agent_details(agent_name, agent_info)

def show_single_agent_details(agent_name: str, agent_info: Dict):
    """Show details for a single agent"""
    st.markdown(f"#### 🤖 {agent_name.replace('_', ' ').title()} Agent")
    
    # Basic agent information
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Status:** {agent_info.get('status', 'unknown').title()}")
        st.info(f"**GPU Assignment:** {agent_info.get('gpu_id', 'API-based')}")
        st.info(f"**Engine Context:** {agent_info.get('engine_context', 'unknown')}")
    
    with col2:
        st.info(f"**Database Path:** {agent_info.get('db_path', 'unknown')}")
        st.info(f"**Coordinator Reference:** {'✅' if agent_info.get('coordinator_ref') else '❌'}")
        created_time = agent_info.get('created_time', 'unknown')
        st.info(f"**Created:** {created_time}")
    
    # Agent capabilities
    methods = agent_info.get('methods_available', [])
    if methods:
        st.markdown("##### 🛠️ Available Methods")
        
        # Filter to show main methods (exclude common Python methods)
        main_methods = [m for m in methods if not m.startswith('get_') or m in ['get_stats', 'get_health']]
        main_methods = [m for m in main_methods if m not in ['cleanup', 'initialize']]
        
        if main_methods:
            # Show methods in columns
            method_cols = st.columns(min(len(main_methods), 3))
            for i, method in enumerate(main_methods):
                with method_cols[i % 3]:
                    st.code(f"{method}()", language="python")
        else:
            st.caption("Standard agent methods available")
    
    # Agent statistics if available
    stats = agent_info.get('stats')
    if stats and stats != 'unavailable':
        st.markdown("##### 📊 Agent Statistics")
        if isinstance(stats, dict):
            for key, value in stats.items():
                st.metric(key.replace('_', ' ').title(), value)
        else:
            st.text(str(stats))
    
    # Error information if agent has issues
    if 'error' in agent_info:
        st.markdown("##### ❌ Error Information")
        st.error(agent_info['error'])

def show_agent_database_status():
    """Show database status for agent operations"""
    st.markdown("### 🗄️ Agent Database Status")
    
    agent_status = get_agent_status_info()
    db_status = agent_status.get('database_status', {})
    
    if not db_status:
        st.warning("Database status unavailable")
        return
    
    # Database overview
    status = db_status.get('status', 'unknown')
    
    if status == 'connected':
        st.success("✅ Database Connected")
    elif status == 'missing':
        st.error("❌ Database File Missing")
    else:
        st.error(f"❌ Database Error: {db_status.get('error', 'unknown')}")
    
    # Database details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        db_path = db_status.get('path', 'unknown')
        st.info(f"**Path:** {os.path.basename(db_path) if db_path != 'unknown' else 'unknown'}")
    
    with col2:
        size_mb = db_status.get('size_mb', 0)
        st.info(f"**Size:** {size_mb:.2f} MB")
    
    with col3:
        tables = db_status.get('tables', {})
        available_tables = sum(1 for table_info in tables.values() if table_info.get('status') == 'available')
        st.info(f"**Tables:** {available_tables}/{len(tables)} available")
    
    # Table status details
    if tables:
        st.markdown("#### 📋 Table Status")
        
        for table_name, table_info in tables.items():
            table_status = table_info.get('status', 'unknown')
            record_count = table_info.get('record_count', 0)
            
            if table_status == 'available':
                st.success(f"✅ **{table_name}**: {record_count} records")
            else:
                st.error(f"❌ **{table_name}**: Missing")

def show_agent_performance_metrics():
    """Show agent performance metrics and usage patterns"""
    st.markdown("### 📈 Agent Performance Metrics")
    
    # Get processing history from session state
    processing_history = st.session_state.get('processing_history', [])
    chat_history = st.session_state.get('chat_history', [])
    
    if not processing_history and not chat_history:
        st.info("No performance data available yet")
        return
    
    # Processing metrics
    if processing_history:
        st.markdown("#### 📊 Processing Activity")
        
        df_processing = pd.DataFrame(processing_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_files = df_processing['files_count'].sum() if 'files_count' in df_processing.columns else 0
            st.metric("Total Files Processed", total_files)
        
        with col2:
            avg_processing_time = df_processing['processing_time'].mean() if 'processing_time' in df_processing.columns else 0
            st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
        
        with col3:
            success_rate = (df_processing['status'] == 'success').mean() * 100 if 'status' in df_processing.columns else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Processing timeline
        if len(df_processing) > 1:
            fig_timeline = px.line(df_processing, x="timestamp", y="files_count",
                                 title="File Processing Activity Over Time",
                                 markers=True)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Chat activity metrics
    if chat_history:
        st.markdown("#### 💬 Chat Agent Activity")
        
        chat_count = len(chat_history)
        user_messages = sum(1 for msg in chat_history if msg.get('role') == 'user')
        assistant_messages = sum(1 for msg in chat_history if msg.get('role') == 'assistant')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Messages", chat_count)
        
        with col2:
            st.metric("User Queries", user_messages)
        
        with col3:
            st.metric("Agent Responses", assistant_messages)

def run_agent_health_check():
    """Run comprehensive health check on all agents"""
    st.markdown("### 🏥 Agent Health Check")
    
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    with st.spinner("Running agent health check..."):
        try:
            agent_status = get_agent_status_info()
            
            if 'error' in agent_status:
                st.error(f"Health check failed: {agent_status['error']}")
                return
            
            # Check coordinator health
            coordinator_info = agent_status.get('coordinator_info', {})
            st.success(f"✅ Coordinator: {coordinator_info.get('type', 'unknown')} - Active")
            
            # Check each agent
            agents = agent_status.get('agents', {})
            
            if agents:
                st.markdown("#### Agent Status:")
                for agent_type, agent_info in agents.items():
                    status = agent_info.get('status', 'unknown')
                    if status == 'active':
                        st.success(f"✅ {agent_type.replace('_', ' ').title()} Agent: Healthy")
                    else:
                        st.error(f"❌ {agent_type.replace('_', ' ').title()} Agent: {status}")
                        if 'error' in agent_info:
                            st.caption(f"   Error: {agent_info['error']}")
            else:
                st.info("ℹ️ No agents currently instantiated (will be created on demand)")
            
            # Check database
            db_status = agent_status.get('database_status', {})
            db_health = db_status.get('status', 'unknown')
            
            if db_health == 'connected':
                st.success("✅ Database: Connected and accessible")
                
                tables = db_status.get('tables', {})
                available_tables = sum(1 for table_info in tables.values() if table_info.get('status') == 'available')
                total_tables = len(tables)
                
                if available_tables == total_tables:
                    st.success(f"✅ Database Tables: All {total_tables} tables available")
                else:
                    st.warning(f"⚠️ Database Tables: {available_tables}/{total_tables} available")
            else:
                st.error(f"❌ Database: {db_health}")
            
            # Overall health summary
            st.markdown("#### 📋 Health Summary")
            
            healthy_components = []
            if coordinator_info:
                healthy_components.append("Coordinator")
            if agents:
                healthy_agents = sum(1 for agent in agents.values() if agent.get('status') == 'active')
                healthy_components.append(f"Agents ({healthy_agents}/{len(agents)})")
            if db_health == 'connected':
                healthy_components.append("Database")
            
            if len(healthy_components) >= 2:
                st.success(f"✅ Overall Status: Healthy ({', '.join(healthy_components)})")
            else:
                st.warning(f"⚠️ Overall Status: Limited functionality ({', '.join(healthy_components)})")
                
        except Exception as e:
            st.error(f"Health check failed: {str(e)}")

def show_agent_status_page():
    """Dedicated agent status monitoring page"""
    st.markdown('<div class="sub-header">🤖 Agent Status & Performance Monitoring</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("System not initialized. Please initialize the system in System Health tab.")
        return
    
    # Agent status tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔧 Detailed Status", 
        "🗄️ Database", 
        "📈 Performance",
        "🏥 Health Check"
    ])
    
    with tab1:
        show_agent_status_overview()
    
    with tab2:
        show_detailed_agent_status()
    
    with tab3:
        show_agent_database_status()
    
    with tab4:
        show_agent_performance_metrics()
    
    with tab5:
        run_agent_health_check()

# Additional function to add agent status to existing displays
def add_agent_status_to_server_display():
    """Add agent status information to existing server displays"""
    if st.session_state.coordinator:
        with st.expander("🤖 Agent Status Summary", expanded=False):
            agent_status = get_agent_status_info()
            
            if 'error' not in agent_status:
                coordinator_info = agent_status.get('coordinator_info', {})
                agents = agent_status.get('agents', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Active Agents", coordinator_info.get('active_agents', 0))
                
                with col2:
                    available_types = len(coordinator_info.get('available_agent_types', []))
                    st.metric("Available Types", available_types)
                
                with col3:
                    db_status = agent_status.get('database_status', {}).get('status', 'unknown')
                    st.metric("Database", "✅" if db_status == 'connected' else "❌")
                
                # Quick agent health
                if agents:
                    healthy_agents = sum(1 for agent in agents.values() if agent.get('status') == 'active')
                    if healthy_agents == len(agents):
                        st.success(f"All {len(agents)} instantiated agents are healthy")
                    else:
                        st.warning(f"{healthy_agents}/{len(agents)} agents are healthy")
                else:
                    st.info("Agents will be created on-demand when needed")
            else:
                st.error("Agent status unavailable")

def show_sidebar_quick_actions():
    """Quick actions in sidebar"""
    if st.button("🔄 Refresh All"):
        st.rerun()
    
    if st.session_state.coordinator:
        if st.button("🏥 Quick Health Check"):
            with st.spinner("Running checks..."):
                # Run both GPU and agent health checks
                run_gpu_health_check()
                run_agent_health_check()
    
    if st.button("📊 System Report"):
        st.info("Navigate to 📊 Full Monitoring for complete report")

def show_complete_monitoring_dashboard():
    """Complete monitoring dashboard combining all views"""
    st.markdown('<div class="sub-header">📊 Complete System Monitoring Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("System not initialized")
        return
    
    # Overall system health
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🌐 System Health")
        try:
            health = st.session_state.coordinator.get_health_status()
            if health.get("status") == "healthy":
                st.success("✅ System Healthy")
            else:
                st.error("❌ System Issues")
        except:
            st.warning("⚠️ Status Unknown")
    
    with col2:
        st.markdown("#### 🎮 GPU Status") 
        try:
            detailed_status = get_detailed_server_status()
            total_gpus = sum(len(stats.get('gpu_info', {})) for stats in detailed_status.values() if 'error' not in stats)
            healthy_gpus = sum(
                len(stats.get('gpu_info', {})) 
                for stats in detailed_status.values() 
                if 'error' not in stats and stats.get('status') == 'healthy'
            )
            if healthy_gpus == total_gpus and total_gpus > 0:
                st.success(f"✅ {total_gpus} GPUs Healthy")
            else:
                st.warning(f"⚠️ {healthy_gpus}/{total_gpus} GPUs")
        except:
            st.error("❌ GPU Status Unknown")
    
    with col3:
        st.markdown("#### 🤖 Agent Status")
        try:
            agent_status = get_agent_status_info()
            if 'error' not in agent_status:
                active_agents = agent_status.get('coordinator_info', {}).get('active_agents', 0)
                st.success(f"✅ {active_agents} Agents Active")
            else:
                st.error("❌ Agent Status Error")
        except:
            st.error("❌ Agent Status Unknown")
    
    # Tabbed detailed view
    tab1, tab2, tab3 = st.tabs(["🖥️ Infrastructure", "🤖 Applications", "📈 Performance"])
    
    with tab1:
        st.markdown("### Server & GPU Infrastructure")
        update_server_status_display()
    
    with tab2:
        st.markdown("### Agent & Database Status")
        show_agent_status_overview()
        st.markdown("---")
        show_agent_database_status()
    
    with tab3:
        st.markdown("### Performance Analytics")
        show_gpu_performance_analytics()
        st.markdown("---")
        show_agent_performance_metrics()

def show_dedicated_gpu_monitoring_page():
    """Dedicated GPU monitoring page"""
    st.markdown('<div class="sub-header">🎮 Dedicated GPU Monitoring Center</div>', unsafe_allow_html=True)
    
    if not st.session_state.coordinator:
        st.error("System not initialized. Please initialize the system in System Health tab.")
        return
    
    # Real-time controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False, help="Auto-refresh GPU metrics")
    
    with col2:
        refresh_interval = st.selectbox("Refresh Interval", [5, 10, 30, 60], index=1, help="Seconds between refreshes")
    
    with col3:
        if st.button("🔄 Refresh Now", help="Manually refresh all GPU data"):
            st.rerun()
    
    # GPU monitoring tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-time Metrics", 
        "🎯 Allocation Optimizer", 
        "⚖️ Load Balancing", 
        "🏥 Health Monitor",
        "📈 Performance Analytics"
    ])
    
    with tab1:
        show_realtime_gpu_monitoring()
    
    with tab2:
        show_gpu_allocation_optimizer()
    
    with tab3:
        show_gpu_load_balancing_insights()
    
    with tab4:
        show_gpu_health_monitoring()
    
    with tab5:
        show_gpu_performance_analytics()
    
    # Auto-refresh logic (simple implementation)
    if auto_refresh:
        st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds")
        # Note: For true auto-refresh, you'd need to use st.empty() containers
        # and update them in a loop, or use custom components

def show_gpu_health_monitoring():
    """Dedicated GPU health monitoring section"""
    st.markdown("### 🏥 GPU Health Monitoring")
    
    detailed_status = get_detailed_server_status()
    
    if not detailed_status:
        st.warning("No GPU data available")
        return
    
    # Health summary
    total_gpus = 0
    healthy_gpus = 0
    warning_gpus = 0
    critical_gpus = 0
    
    health_details = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
            
        gpu_info = stats.get('gpu_info', {})
        server_healthy = stats.get('status') == 'healthy'
        
        for gpu_name, gpu_data in gpu_info.items():
            total_gpus += 1
            
            utilization = gpu_data.get('utilization', 0)
            memory_total = gpu_data.get('total_memory', 0)
            memory_used = gpu_data.get('memory_cached', 0)
            memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
            
            # Determine health status
            health_status = "healthy"
            health_message = "Operating normally"
            issues = []
            
            if memory_percent > 95:
                health_status = "critical"
                issues.append(f"Critical memory usage: {memory_percent:.1f}%")
                critical_gpus += 1
            elif memory_percent > 85:
                health_status = "warning"
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                warning_gpus += 1
            elif utilization > 95:
                health_status = "warning"
                issues.append(f"Very high utilization: {utilization:.1f}%")
                warning_gpus += 1
            elif not server_healthy:
                health_status = "warning"
                issues.append("Server unhealthy")
                warning_gpus += 1
            else:
                healthy_gpus += 1
            
            if issues:
                health_message = "; ".join(issues)
            
            health_details.append({
                'server': server_name,
                'gpu': gpu_name,
                'status': health_status,
                'message': health_message,
                'utilization': utilization,
                'memory_percent': memory_percent,
                'gpu_model': gpu_data.get('name', 'Unknown')
            })
    
    # Health overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total GPUs", total_gpus)
    
    with col2:
        st.metric("Healthy", healthy_gpus, delta=None)
        if healthy_gpus == total_gpus:
            st.success("All GPUs healthy")
    
    with col3:
        st.metric("Warnings", warning_gpus, delta=None)
        if warning_gpus > 0:
            st.warning(f"{warning_gpus} GPUs need attention")
    
    with col4:
        st.metric("Critical", critical_gpus, delta=None)
        if critical_gpus > 0:
            st.error(f"{critical_gpus} GPUs in critical state")
    
    # Detailed health status
    st.markdown("#### 🔍 Detailed Health Status")
    
    for detail in health_details:
        if detail['status'] == 'critical':
            st.error(f"🚨 **{detail['server']} ({detail['gpu']})**: {detail['message']}")
        elif detail['status'] == 'warning':
            st.warning(f"⚠️ **{detail['server']} ({detail['gpu']})**: {detail['message']}")
        else:
            st.success(f"✅ **{detail['server']} ({detail['gpu']})**: {detail['message']}")
    
    # Health trends (if you want to track over time)
    st.markdown("#### 📈 Health Trends")
    st.info("Health trend tracking would require storing historical data over time")

def show_gpu_performance_analytics():
    """GPU performance analytics and insights"""
    st.markdown("### 📈 GPU Performance Analytics")
    
    detailed_status = get_detailed_server_status()
    
    if not detailed_status:
        st.warning("No performance data available")
        return
    
    # Collect performance data
    performance_data = []
    
    for server_name, stats in detailed_status.items():
        if 'error' in stats:
            continue
            
        gpu_info = stats.get('gpu_info', {})
        active_requests = stats.get('active_requests', 0)
        total_requests = stats.get('total_requests', 0)
        success_rate = stats.get('success_rate', 0)
        avg_latency = stats.get('average_latency', 0)
        uptime = stats.get('uptime', 0)
        
        for gpu_name, gpu_data in gpu_info.items():
            utilization = gpu_data.get('utilization', 0)
            memory_total = gpu_data.get('total_memory', 0)
            memory_used = gpu_data.get('memory_cached', 0)
            memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
            
            # Calculate performance metrics
            requests_per_hour = (total_requests / max(uptime, 1)) * 3600 if uptime > 0 else 0
            efficiency_score = (success_rate * utilization / 100) if utilization > 0 else 0
            
            performance_data.append({
                'server': server_name,
                'gpu': gpu_name,
                'utilization': utilization,
                'memory_percent': memory_percent,
                'active_requests': active_requests,
                'requests_per_hour': requests_per_hour,
                'success_rate': success_rate,
                'avg_latency': avg_latency,
                'efficiency_score': efficiency_score,
                'memory_gb': memory_total / (1024**3) if memory_total > 0 else 0
            })
    
    if not performance_data:
        st.warning("No performance data to analyze")
        return
    
    df_perf = pd.DataFrame(performance_data)
    
    # Performance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_utilization = df_perf['utilization'].mean()
        st.metric("Avg GPU Utilization", f"{avg_utilization:.1f}%")
    
    with col2:
        avg_memory = df_perf['memory_percent'].mean()
        st.metric("Avg Memory Usage", f"{avg_memory:.1f}%")
    
    with col3:
        total_requests_hour = df_perf['requests_per_hour'].sum()
        st.metric("Total Req/Hour", f"{total_requests_hour:.0f}")
    
    with col4:
        avg_efficiency = df_perf['efficiency_score'].mean()
        st.metric("Avg Efficiency", f"{avg_efficiency:.2f}")
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Utilization vs Efficiency
        fig_eff = px.scatter(df_perf, x="utilization", y="efficiency_score",
                           color="server", size="memory_gb",
                           title="GPU Utilization vs Efficiency",
                           labels={
                               "utilization": "GPU Utilization (%)",
                               "efficiency_score": "Efficiency Score"
                           })
        st.plotly_chart(fig_eff, use_container_width=True)
    
    with col2:
        # Memory vs Performance
        fig_mem = px.scatter(df_perf, x="memory_percent", y="requests_per_hour",
                           color="avg_latency", size="utilization",
                           title="Memory Usage vs Request Rate",
                           labels={
                               "memory_percent": "Memory Usage (%)",
                               "requests_per_hour": "Requests per Hour"
                           })
        st.plotly_chart(fig_mem, use_container_width=True)
    
    # Performance insights
    st.markdown("#### 💡 Performance Insights")
    
    # Find best and worst performers
    best_performer = df_perf.loc[df_perf['efficiency_score'].idxmax()]
    worst_performer = df_perf.loc[df_perf['efficiency_score'].idxmin()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("🏆 Best Performer")
        st.write(f"**Server**: {best_performer['server']}")
        st.write(f"**GPU**: {best_performer['gpu']}")
        st.write(f"**Efficiency**: {best_performer['efficiency_score']:.2f}")
        st.write(f"**Utilization**: {best_performer['utilization']:.1f}%")
    
    with col2:
        st.error("📉 Needs Improvement")
        st.write(f"**Server**: {worst_performer['server']}")
        st.write(f"**GPU**: {worst_performer['gpu']}")
        st.write(f"**Efficiency**: {worst_performer['efficiency_score']:.2f}")
        st.write(f"**Utilization**: {worst_performer['utilization']:.1f}%")
    
    # Performance recommendations
    st.markdown("#### 🎯 Performance Recommendations")
    
    recommendations = []
    
    # Check for underutilized GPUs
    underutilized = df_perf[df_perf['utilization'] < 20]
    if not underutilized.empty:
        recommendations.append("🔄 Consider redistributing load from underutilized GPUs")
    
    # Check for memory pressure
    high_memory = df_perf[df_perf['memory_percent'] > 85]
    if not high_memory.empty:
        recommendations.append("💾 Monitor memory usage on high-memory servers")
    
    # Check for efficiency issues
    low_efficiency = df_perf[df_perf['efficiency_score'] < 0.1]
    if not low_efficiency.empty:
        recommendations.append("⚡ Investigate efficiency issues on low-performing GPUs")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("✅ All GPUs are performing optimally")

def show_enhanced_footer():
    """Enhanced footer with GPU system information"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌐 Opulence Enhanced API Research Agent**")
        st.markdown("Advanced GPU monitoring & load balancing")
        if st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                available_servers = health.get('available_servers', 0)
                total_servers = health.get('total_servers', 0)
                st.markdown(f"API Servers: {available_servers}/{total_servers} available")
                
                # Quick GPU count
                detailed_status = get_detailed_server_status()
                total_gpus = sum(len(stats.get('gpu_info', {})) for stats in detailed_status.values() if 'error' not in stats)
                st.markdown(f"GPUs Monitored: {total_gpus}")
            except:
                st.markdown("Status: Monitoring active")
    
    with col2:
        st.markdown("**📊 Current Session**")
        st.markdown(f"Chat Messages: {len(st.session_state.chat_history)}")
        st.markdown(f"Files Processed: {len(st.session_state.processing_history)}")
        
        # GPU utilization summary
        if st.session_state.coordinator:
            try:
                detailed_status = get_detailed_server_status()
                gpu_utils = []
                for stats in detailed_status.values():
                    if 'error' not in stats:
                        gpu_info = stats.get('gpu_info', {})
                        for gpu_data in gpu_info.values():
                            gpu_utils.append(gpu_data.get('utilization', 0))
                
                if gpu_utils:
                    avg_util = sum(gpu_utils) / len(gpu_utils)
                    st.markdown(f"Avg GPU Utilization: {avg_util:.1f}%")
            except:
                pass
    
    with col3:
        st.markdown("**🕐 System Information**")
        st.markdown(f"Local Time: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.session_state.coordinator:
            try:
                health = st.session_state.coordinator.get_health_status()
                uptime = health.get('uptime_seconds', 0)
                st.markdown(f"System Uptime: {uptime:.0f}s")
                
                coordinator_type = health.get('coordinator_type', 'unknown')
                st.markdown(f"Mode: {coordinator_type.replace('_', ' ').title()}")
            except:
                st.markdown("Mode: API-Based")
# Integration Updates for streamlit_app.py
# Replace the original functions with these enhanced versions

# 1. Replace the original show_server_status function
def show_server_status():
    """Enhanced server status display with comprehensive GPU monitoring"""
    update_server_status_display()

# 2. Replace the original show_system_health function  
def show_system_health():
    """Enhanced system health page"""
    show_enhanced_system_health()

# 3. Replace the original show_dashboard function
def show_dashboard():
    """Enhanced dashboard with GPU monitoring"""
    show_enhanced_dashboard()

# 4. Replace the original main function
def main():
    """Enhanced main application function"""
    main_enhanced()

# 5. Enhanced coordinator health checking
async def enhanced_init_api_coordinator():
    """Enhanced coordinator initialization with GPU validation"""
    if not COORDINATOR_AVAILABLE:
        return False
        
    if st.session_state.coordinator is None:
        try:
            # Create API coordinator with configured servers
            coordinator = create_api_coordinator_from_config(
                model_servers=st.session_state.model_servers,
                load_balancing_strategy="least_busy"
            )
            
            # Initialize the coordinator
            await coordinator.initialize()
            
            # Validate GPU endpoints
            gpu_validation_success = True
            for server_config in st.session_state.model_servers:
                try:
                    endpoint = server_config.get('endpoint')
                    if endpoint:
                        response = requests.get(f"{endpoint}/health", timeout=5)
                        if response.status_code != 200:
                            st.warning(f"Server {server_config.get('name', 'unknown')} health check failed")
                            gpu_validation_success = False
                except:
                    st.warning(f"Cannot reach server {server_config.get('name', 'unknown')}")
                    gpu_validation_success = False
            
            if not gpu_validation_success:
                st.warning("Some GPU servers are not responding, but coordinator initialized")
            
            st.session_state.coordinator = coordinator
            st.session_state.initialization_status = "completed"
            return True
            
        except Exception as e:
            st.session_state.initialization_status = f"error: {str(e)}"
            return False
    return True

# 6. Enhanced model server configuration with validation
def enhanced_configure_model_servers():
    """Enhanced model server configuration with GPU validation"""
    st.markdown("### 🌐 Configure Model Servers with GPU Monitoring")
    
    # Server configuration form with GPU testing
    with st.form("server_config"):
        st.markdown("#### Add/Edit Server")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            server_name = st.text_input("Server Name", value="gpu_1")
        
        with col2:
            endpoint = st.text_input("Endpoint", value="http://localhost:8000")
        
        with col3:
            gpu_id = st.number_input("GPU ID", min_value=0, value=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_requests = st.number_input("Max Concurrent Requests", min_value=1, value=10)
        
        with col2:
            timeout = st.number_input("Timeout (seconds)", min_value=30, value=300)
        
        # Test connection option
        test_connection = st.checkbox("Test connection before adding", value=True)
        
        if st.form_submit_button("Add/Update Server"):
            new_server = {
                "name": server_name,
                "endpoint": endpoint,
                "gpu_id": gpu_id,
                "max_concurrent_requests": max_requests,
                "timeout": timeout
            }
            
            # Test connection if requested
            connection_ok = True
            if test_connection:
                try:
                    with st.spinner(f"Testing connection to {endpoint}..."):
                        response = requests.get(f"{endpoint}/health", timeout=5)
                        if response.status_code == 200:
                            st.success(f"✅ Connection to {server_name} successful")
                            
                            # Try to get GPU info
                            try:
                                status_response = requests.get(f"{endpoint}/status", timeout=5)
                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    gpu_info = status_data.get('gpu_info', {})
                                    if gpu_info:
                                        gpu_count = len(gpu_info)
                                        st.info(f"🎮 Detected {gpu_count} GPU(s) on {server_name}")
                                    else:
                                        st.warning(f"⚠️ No GPU information available from {server_name}")
                            except:
                                st.warning(f"⚠️ Could not retrieve GPU status from {server_name}")
                        else:
                            st.error(f"❌ Connection failed: HTTP {response.status_code}")
                            connection_ok = False
                except requests.RequestException as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    connection_ok = False
            
            if connection_ok or not test_connection:
                # Check if server already exists
                existing_names = [s.get('name', '') for s in st.session_state.model_servers]
                if server_name in existing_names:
                    # Update existing server
                    for i, server in enumerate(st.session_state.model_servers):
                        if server.get('name') == server_name:
                            st.session_state.model_servers[i] = new_server
                            break
                    st.success(f"Updated server: {server_name}")
                else:
                    # Add new server
                    st.session_state.model_servers.append(new_server)
                    st.success(f"Added new server: {server_name}")
                
                st.rerun()
    
    # Current servers display with enhanced info
    st.markdown("#### Current Servers")
    if st.session_state.model_servers:
        for i, server in enumerate(st.session_state.model_servers):
            with st.expander(f"🖥️ {server.get('name', f'Server {i+1}')}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Endpoint:** {server.get('endpoint', 'N/A')}")
                    st.write(f"**GPU ID:** {server.get('gpu_id', 'N/A')}")
                
                with col2:
                    st.write(f"**Max Requests:** {server.get('max_concurrent_requests', 10)}")
                    st.write(f"**Timeout:** {server.get('timeout', 300)}s")
                
                # Quick status check
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"🔍 Test {server.get('name')}", key=f"test_{i}"):
                        try:
                            endpoint = server.get('endpoint')
                            response = requests.get(f"{endpoint}/health", timeout=5)
                            if response.status_code == 200:
                                st.success("✅ Server responding")
                            else:
                                st.error(f"❌ HTTP {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
                
                with col2:
                    if st.button(f"🎮 GPU Info {server.get('name')}", key=f"gpu_{i}"):
                        try:
                            endpoint = server.get('endpoint')
                            response = requests.get(f"{endpoint}/status", timeout=5)
                            if response.status_code == 200:
                                status_data = response.json()
                                gpu_info = status_data.get('gpu_info', {})
                                if gpu_info:
                                    st.json(gpu_info)
                                else:
                                    st.info("No GPU information available")
                            else:
                                st.error(f"❌ HTTP {response.status_code}")
                        except Exception as e:
                            st.error(f"❌ {str(e)}")
                
                with col3:
                    if st.button(f"🗑️ Remove {server.get('name')}", key=f"remove_{i}"):
                        st.session_state.model_servers.pop(i)
                        st.rerun()
    else:
        st.info("No servers configured. Add servers above to get started.")

# 7. Enhanced process_chat_query with GPU context
def enhanced_process_chat_query(query: str) -> Dict[str, Any]:
    """Enhanced chat query processing with GPU context"""
    if not COORDINATOR_AVAILABLE:
        return {
            "response": "❌ API Coordinator not available. Please check the import error in debug mode.",
            "response_type": "error",
            "suggestions": []
        }
    
    if not st.session_state.coordinator:
        return {
            "response": "❌ System not initialized. Please check system health.",
            "response_type": "error",
            "suggestions": ["Initialize system in System Health tab"]
        }
    
    try:
        # Get conversation history for context
        conversation_history = st.session_state.chat_history[-5:] if st.session_state.chat_history else []
        
        # Add GPU context to query if relevant
        gpu_keywords = ['gpu', 'utilization', 'memory', 'performance', 'server', 'load']
        if any(keyword in query.lower() for keyword in gpu_keywords):
            # Add current GPU status as context
            try:
                detailed_status = get_detailed_server_status()
                gpu_context = f"\n\nCurrent GPU Status: {len([s for s in detailed_status.values() if 'error' not in s])} servers available"
                query_with_context = query + gpu_context
            except:
                query_with_context = query
        else:
            query_with_context = query
        
        # Process with the API coordinator
        result = safe_run_async(
            st.session_state.coordinator.process_chat_query(query_with_context, conversation_history)
        )
        
        if isinstance(result, dict):
            # Add GPU server info to response
            if "servers_used" not in result:
                try:
                    health = st.session_state.coordinator.get_health_status()
                    result["servers_used"] = list(health.get('server_stats', {}).keys())
                except:
                    result["servers_used"] = "unknown"
            
            # Add GPU utilization info if available
            try:
                detailed_status = get_detailed_server_status()
                gpu_utils = []
                for stats in detailed_status.values():
                    if 'error' not in stats:
                        gpu_info = stats.get('gpu_info', {})
                        for gpu_data in gpu_info.values():
                            gpu_utils.append(gpu_data.get('utilization', 0))
                
                if gpu_utils:
                    avg_util = sum(gpu_utils) / len(gpu_utils)
                    result["gpu_utilization"] = f"{avg_util:.1f}%"
            except:
                pass
            
            return result
        else:
            return {
                "response": str(result),
                "response_type": "general",
                "suggestions": [],
                "servers_used": "unknown"
            }
    
    except Exception as e:
        return {
            "response": f"❌ Error processing query: {str(e)}",
            "response_type": "error",
            "suggestions": ["Try rephrasing your question", "Check system status"],
            "servers_used": "unknown"
        }

# 8. Replace original functions in main execution
# Add this at the bottom of your streamlit_app.py file

# Replace original function calls
configure_model_servers = enhanced_configure_model_servers
init_api_coordinator = enhanced_init_api_coordinator
process_chat_query = enhanced_process_chat_query

# Update the main execution
if __name__ == "__main__":
    try:
        main_enhanced()  # Use enhanced main function
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)
        
        # Emergency debug mode
        st.markdown("### Emergency Debug Info")
        st.json({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "session_state": dict(st.session_state),
            "coordinator_type": "api_based_enhanced",
            "gpu_monitoring_enabled": True
        })

# 9. Enhanced Coordinator Load Balancing Updates
# Add these enhancements to the coordinator file (api_opulence_coordinator.py)

class EnhancedLoadBalancingStrategy(Enum):
    """Enhanced load balancing strategies including GPU-aware options"""
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    LEAST_LATENCY = "least_latency"
    LEAST_GPU_UTILIZATION = "least_gpu_utilization"
    LEAST_MEMORY_USAGE = "least_memory_usage"
    BALANCED_GPU_MEMORY = "balanced_gpu_memory"
    RANDOM = "random"

class EnhancedLoadBalancer:
    """Enhanced load balancer with GPU-aware routing"""
    
    def __init__(self, config):
        self.config = config
        self.servers = []
        self.current_index = 0
        self.logger = logging.getLogger(f"{__name__}.EnhancedLoadBalancer")
        
        # Initialize servers
        for server_config in config.model_servers:
            self.servers.append(ModelServer(server_config))
    
    def select_server_gpu_aware(self) -> Optional[ModelServer]:
        """Select server based on GPU-aware load balancing"""
        available_servers = self.get_available_servers()
        
        if not available_servers:
            return None
        
        strategy = self.config.load_balancing_strategy
        
        if strategy == EnhancedLoadBalancingStrategy.LEAST_GPU_UTILIZATION:
            return self._select_by_gpu_utilization(available_servers)
        elif strategy == EnhancedLoadBalancingStrategy.LEAST_MEMORY_USAGE:
            return self._select_by_memory_usage(available_servers)
        elif strategy == EnhancedLoadBalancingStrategy.BALANCED_GPU_MEMORY:
            return self._select_by_gpu_memory_balance(available_servers)
        else:
            # Fall back to original strategies
            return self.select_server()
    
    def _select_by_gpu_utilization(self, servers):
        """Select server with lowest GPU utilization"""
        try:
            gpu_utilizations = {}
            for server in servers:
                try:
                    # Get real-time GPU utilization
                    response = requests.get(f"{server.config.endpoint}/metrics", timeout=2)
                    if response.status_code == 200:
                        metrics = response.json()
                        gpu_utilizations[server] = metrics.get('gpu_utilization', 100)
                    else:
                        gpu_utilizations[server] = 100  # Assume high if can't get metrics
                except:
                    gpu_utilizations[server] = 100  # Assume high if can't connect
            
            return min(gpu_utilizations.keys(), key=lambda s: gpu_utilizations[s])
        except:
            return servers[0]  # Fallback
    
    def _select_by_memory_usage(self, servers):
        """Select server with lowest memory usage"""
        try:
            memory_usages = {}
            for server in servers:
                try:
                    response = requests.get(f"{server.config.endpoint}/status", timeout=2)
                    if response.status_code == 200:
                        status = response.json()
                        gpu_info = status.get('gpu_info', {})
                        
                        # Calculate average memory usage across GPUs
                        memory_percentages = []
                        for gpu_data in gpu_info.values():
                            total_mem = gpu_data.get('total_memory', 0)
                            used_mem = gpu_data.get('memory_cached', 0)
                            if total_mem > 0:
                                memory_percentages.append((used_mem / total_mem) * 100)
                        
                        avg_memory = sum(memory_percentages) / len(memory_percentages) if memory_percentages else 100
                        memory_usages[server] = avg_memory
                    else:
                        memory_usages[server] = 100
                except:
                    memory_usages[server] = 100
            
            return min(memory_usages.keys(), key=lambda s: memory_usages[s])
        except:
            return servers[0]
    
    def _select_by_gpu_memory_balance(self, servers):
        """Select server with best GPU utilization to memory ratio"""
        try:
            balance_scores = {}
            for server in servers:
                try:
                    response = requests.get(f"{server.config.endpoint}/status", timeout=2)
                    if response.status_code == 200:
                        status = response.json()
                        gpu_info = status.get('gpu_info', {})
                        
                        utilizations = []
                        memory_percentages = []
                        
                        for gpu_data in gpu_info.values():
                            utilizations.append(gpu_data.get('utilization', 0))
                            total_mem = gpu_data.get('total_memory', 0)
                            used_mem = gpu_data.get('memory_cached', 0)
                            if total_mem > 0:
                                memory_percentages.append((used_mem / total_mem) * 100)
                        
                        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0
                        avg_memory = sum(memory_percentages) / len(memory_percentages) if memory_percentages else 100
                        
                        # Balance score: prefer higher utilization but lower memory usage
                        # Scale utilization positively and memory usage negatively
                        balance_score = avg_util - (avg_memory * 0.5)
                        balance_scores[server] = balance_score
                    else:
                        balance_scores[server] = -100  # Low score for unreachable servers
                except:
                    balance_scores[server] = -100
            
            return max(balance_scores.keys(), key=lambda s: balance_scores[s])
        except:
            return servers[0]

# 10. Enhanced Model Server with Better GPU Monitoring
def enhanced_model_server_gpu_info():
    """Enhanced GPU information collection for model server"""
    
    def _collect_enhanced_gpu_info(self):
        """Enhanced GPU information collection with more metrics"""
        self.gpu_info = {}
        for gpu_id in self.config.gpu_ids:
            try:
                # Basic GPU properties
                properties = torch.cuda.get_device_properties(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id)
                memory_cached = torch.cuda.memory_reserved(gpu_id)
                
                # Try to get more detailed metrics if available
                try:
                    import nvidia_ml_py3 as nvml
                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    
                    # Get real GPU utilization (not just memory)
                    utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu
                    
                    # Get temperature
                    temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    
                    # Get power usage
                    power_usage = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    
                except ImportError:
                    # Fallback to memory-based utilization if nvidia-ml-py3 not available
                    gpu_utilization = (memory_cached / properties.total_memory) * 100
                    temperature = None
                    power_usage = None
                except Exception:
                    gpu_utilization = (memory_cached / properties.total_memory) * 100
                    temperature = None
                    power_usage = None
                
                self.gpu_info[f"gpu_{gpu_id}"] = {
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory": properties.total_memory,
                    "memory_allocated": memory_allocated,
                    "memory_cached": memory_cached,
                    "memory_free": properties.total_memory - memory_cached,
                    "utilization": gpu_utilization,
                    "temperature": temperature,
                    "power_usage": power_usage,
                    "memory_bandwidth": getattr(properties, 'memory_bandwidth', None),
                    "multiprocessor_count": properties.multi_processor_count
                }
            except Exception as e:
                self.logger.warning(f"Failed to collect GPU {gpu_id} info: {str(e)}")
    
    return _collect_enhanced_gpu_info

# 11. Usage Instructions for Integration

"""
INTEGRATION INSTRUCTIONS:

1. **Replace Functions in streamlit_app.py:**
   - Replace `show_server_status()` with the enhanced version
   - Replace `show_system_health()` with the enhanced version  
   - Replace `show_dashboard()` with the enhanced version
   - Replace `main()` with `main_enhanced()`
   - Replace `configure_model_servers()` with `enhanced_configure_model_servers()`
   - Replace `process_chat_query()` with `enhanced_process_chat_query()`

2. **Add New Functions to streamlit_app.py:**
   - Add all functions from Parts 1, 2, and 3
   - Import requests: `import requests`
   - Ensure pandas and plotly are imported

3. **Update api_opulence_coordinator.py:**
   - Add EnhancedLoadBalancingStrategy enum
   - Replace LoadBalancer with EnhancedLoadBalancer
   - Update ModelServerConfig to use enhanced GPU monitoring

4. **Update model_server.py:**
   - Replace _collect_gpu_info method with enhanced version
   - Add nvidia-ml-py3 dependency for better GPU monitoring:
     `pip install nvidia-ml-py3`

5. **Dependencies to Add:**
   ```bash
   pip install nvidia-ml-py3  # For better GPU monitoring
   pip install requests       # For server communication
   pip install plotly         # For enhanced charts (already included)
   ```

6. **Configuration:**
   - Ensure model servers expose /health, /status, and /metrics endpoints
   - Configure proper server endpoints in Streamlit interface
   - Test GPU monitoring endpoints before deployment

7. **Optional Enhancements:**
   - Add real-time auto-refresh using st.empty() containers
   - Implement historical GPU metrics storage
   - Add alerting for GPU threshold violations
   - Create custom Streamlit components for advanced GPU visualization

The enhanced system provides:
✅ Real-time GPU utilization monitoring
✅ GPU memory usage tracking  
✅ GPU-aware load balancing
✅ Comprehensive health monitoring
✅ Performance analytics
✅ Load balancing optimization
✅ Enhanced error handling
✅ Better user interface
"""