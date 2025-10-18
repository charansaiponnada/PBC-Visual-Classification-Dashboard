import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.animation as animation
from io import BytesIO
import json
import math

# Page configuration
st.set_page_config(
    page_title="PBC Visual Classification",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CLASSES ====================

class TreeNode:
    """Represents a node in the decision tree"""
    def __init__(self, node_id, data_indices, depth=0, parent=None):
        self.node_id = node_id
        self.data_indices = data_indices
        self.depth = depth
        self.parent = parent
        self.children = []
        self.split_attribute = None
        self.split_points = []
        self.class_label = None
        self.is_leaf = False
        self.used_attributes = set()
        if parent:
            self.used_attributes = parent.used_attributes.copy()

class Rule:
    """Rule class for PBC"""
    def __init__(self, rule_type, params, predicted_class, node_id=None):
        self.rule_type = rule_type
        self.params = params
        self.predicted_class = predicted_class
        self.node_id = node_id
    
    def matches(self, x, y):
        if self.rule_type == 'circle':
            cx, cy, r = self.params['cx'], self.params['cy'], self.params['r']
            return (x - cx)**2 + (y - cy)**2 <= r**2
        elif self.rule_type == 'rectangle':
            x1, y1, x2, y2 = self.params['x1'], self.params['y1'], self.params['x2'], self.params['y2']
            return x1 <= x <= x2 and y1 <= y <= y2
        elif self.rule_type == 'axis_x':
            threshold = self.params['threshold']
            direction = self.params['direction']
            return x <= threshold if direction == 'left' else x > threshold
        elif self.rule_type == 'axis_y':
            threshold = self.params['threshold']
            direction = self.params['direction']
            return y <= threshold if direction == 'below' else y > threshold
        elif self.rule_type == 'splitline':
            attr = self.params['attribute']
            value = self.params['value']
            direction = self.params['direction']
            # This will be handled differently in actual classification
            return True
        return False

class PBCClassifier:
    """PBC Classifier with tree structure"""
    def __init__(self):
        self.rules = []
        self.tree_root = None
        self.current_node = None
        self.node_counter = 0
        self.history = []
    
    def add_rule(self, rule):
        self.rules.append(rule)
        self.history.append(('add_rule', rule))
    
    def predict(self, X, y_data, default_class=None):
        predictions = []
        for i in range(len(X)):
            x, y = X[i], y_data[i]
            predicted = default_class
            for rule in self.rules:
                if rule.matches(x, y):
                    predicted = rule.predicted_class
                    break
            predictions.append(predicted)
        return predictions
    
    def undo_last_action(self):
        if self.history:
            action = self.history.pop()
            if action[0] == 'add_rule':
                self.rules.pop()
            return True
        return False

# ==================== HELPER FUNCTIONS ====================

def apply_theme(dark_mode):
    """Apply custom CSS theme"""
    if dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .metric-card {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .info-box {
            background-color: #1e293b;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .splitline-active {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #1f2937;
        }
        .metric-card {
            background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .info-box {
            background-color: #f3f4f6;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .splitline-active {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        </style>
        """, unsafe_allow_html=True)

def create_circle_segments_plot(df, selected_features, classes, class_colors):
    """Create proper Circle Segments visualization from PBC paper"""
    fig = go.Figure()
    
    n_features = len(selected_features)
    if n_features == 0:
        return fig
    
    angle_per_feature = 360 / n_features
    
    # Normalize and sort data for each attribute
    for idx, feature in enumerate(selected_features):
        # Sort by feature value
        sorted_data = df.sort_values(by=feature).reset_index(drop=True)
        
        start_angle = idx * angle_per_feature
        end_angle = (idx + 1) * angle_per_feature
        mid_angle = (start_angle + end_angle) / 2
        
        # Calculate segment boundaries
        segment_start_rad = np.radians(start_angle)
        segment_end_rad = np.radians(end_angle)
        
        # Draw segment boundary lines
        for angle in [start_angle, end_angle]:
            angle_rad = np.radians(angle)
            fig.add_trace(go.Scatter(
                x=[0, 1.2*np.cos(angle_rad)],
                y=[0, 1.2*np.sin(angle_rad)],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Arrange data points within segment (line-by-line, orthogonal to segment line)
        n_points = len(sorted_data)
        points_per_line = max(5, int(np.sqrt(n_points)))
        
        for i, row in sorted_data.iterrows():
            # Calculate position within segment
            line_num = i // points_per_line
            pos_in_line = i % points_per_line
            
            # Radius increases with line number
            r = 0.3 + (line_num / (n_points / points_per_line)) * 0.8
            
            # Angle varies across the line
            angle_offset = (pos_in_line / points_per_line - 0.5) * (angle_per_feature * 0.8)
            point_angle = mid_angle + angle_offset
            point_angle_rad = np.radians(point_angle)
            
            x = r * np.cos(point_angle_rad)
            y = r * np.sin(point_angle_rad)
            
            class_label = row['class']
            
            # Add point
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(
                    size=4,
                    color=class_colors.get(class_label, '#gray'),
                    line=dict(width=0)
                ),
                name=class_label,
                showlegend=False,
                hovertemplate=f"<b>{feature}</b>: {row[feature]:.2f}<br><b>Class</b>: {class_label}<extra></extra>"
            ))
        
        # Add attribute label
        label_r = 1.35
        label_x = label_r * np.cos(np.radians(mid_angle))
        label_y = label_r * np.sin(np.radians(mid_angle))
        
        fig.add_annotation(
            x=label_x, y=label_y,
            text=f"<b>{feature}</b>",
            showarrow=False,
            font=dict(size=11, color='white' if st.session_state.dark_mode else 'black'),
            bgcolor='rgba(0,0,0,0.7)' if st.session_state.dark_mode else 'rgba(255,255,255,0.9)',
            borderpad=4
        )
    
    # Add legend manually
    for cls in classes:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=class_colors.get(cls, 'gray')),
            name=cls,
            showlegend=True
        ))
    
    fig.update_layout(
        title="Circle Segments Visualization (PBC Method)",
        xaxis=dict(visible=False, range=[-1.6, 1.6]),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x", scaleratio=1),
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest'
    )
    
    return fig

def create_pbc_tree_visualization(rules, classes, class_colors):
    """Create PBC tree visualization from rules"""
    if not rules:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Show empty root
        root_rect = FancyBboxPatch((4, 4.5), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#94a3b8',
                                  edgecolor='#64748b',
                                  linewidth=2)
        ax.add_patch(root_rect)
        ax.text(5, 4.9, "Root Node\n(No rules yet)", 
                ha='center', va='center', fontsize=10, color='white', weight='bold')
        
        ax.text(5, 2, "Create rules to build your decision tree!", 
                ha='center', va='center', fontsize=11, style='italic', color='gray')
        return fig
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Group rules by class
    rules_by_class = {}
    for rule in rules:
        if rule.predicted_class not in rules_by_class:
            rules_by_class[rule.predicted_class] = []
        rules_by_class[rule.predicted_class].append(rule)
    
    # Draw root
    root_rect = FancyBboxPatch((6, 10), 2, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='#3b82f6',
                              edgecolor='#1e40af',
                              linewidth=3)
    ax.add_patch(root_rect)
    ax.text(7, 10.4, f"All Data\n({len(rules)} rules)", 
            ha='center', va='center', fontsize=11, color='white', weight='bold')
    
    # Draw class branches
    n_classes = len(rules_by_class)
    if n_classes > 0:
        spacing = 12 / (n_classes + 1)
        y_class = 7.5
        
        for i, (cls, cls_rules) in enumerate(sorted(rules_by_class.items())):
            x = spacing * (i + 1)
            
            # Draw connection line from root
            ax.plot([7, x], [10, y_class + 0.8], 'k-', linewidth=2, alpha=0.6)
            
            # Draw class node
            color = class_colors.get(cls, '#22c55e')
            node_width = 2.5
            node_rect = FancyBboxPatch((x - node_width/2, y_class), node_width, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color,
                                      edgecolor='#16a34a',
                                      linewidth=2)
            ax.add_patch(node_rect)
            
            # Class name and rule count
            ax.text(x, y_class + 0.4, f"Class: {cls}\n{len(cls_rules)} rule(s)", 
                   ha='center', va='center', fontsize=10, color='white', weight='bold')
            
            # Draw individual rules below each class
            y_rule_start = y_class - 1.2
            max_rules_show = min(5, len(cls_rules))
            
            for j, rule in enumerate(cls_rules[:max_rules_show]):
                rule_y = y_rule_start - j * 1.0
                
                # Connection line
                ax.plot([x, x], [y_class, rule_y + 0.4], 'k--', linewidth=1, alpha=0.4)
                
                # Determine rule icon and text
                if rule.rule_type == 'circle':
                    icon = '⭕'
                    rule_desc = f"Circle: r={rule.params.get('r', 0):.1f}"
                elif rule.rule_type == 'rectangle':
                    icon = '▭'
                    rule_desc = "Rectangle"
                elif rule.rule_type == 'axis_x':
                    icon = '│'
                    thresh = rule.params.get('threshold', 0)
                    direction = rule.params.get('direction', 'left')
                    rule_desc = f"X {'≤' if direction == 'left' else '>'} {thresh:.2f}"
                elif rule.rule_type == 'axis_y':
                    icon = '─'
                    thresh = rule.params.get('threshold', 0)
                    direction = rule.params.get('direction', 'below')
                    rule_desc = f"Y {'≤' if direction == 'below' else '>'} {thresh:.2f}"
                elif rule.rule_type == 'splitline':
                    icon = '✂️'
                    attr = rule.params.get('attribute', 'attr')
                    val = rule.params.get('value', 0)
                    rule_desc = f"{attr[:8]} {val:.2f}"
                else:
                    icon = '●'
                    rule_desc = rule.rule_type
                
                # Rule box
                rule_width = 2.0
                rule_rect = FancyBboxPatch((x - rule_width/2, rule_y), rule_width, 0.4,
                                          boxstyle="round,pad=0.05",
                                          facecolor='#fbbf24',
                                          edgecolor='#f59e0b',
                                          linewidth=1.5)
                ax.add_patch(rule_rect)
                ax.text(x, rule_y + 0.2, f"{icon} {rule_desc}", 
                       ha='center', va='center', fontsize=8, color='#1f2937', weight='bold')
            
            # Show "+X more" if there are additional rules
            if len(cls_rules) > max_rules_show:
                extra = len(cls_rules) - max_rules_show
                ax.text(x, rule_y - 0.6, f"+{extra} more rule{'s' if extra > 1 else ''}", 
                       ha='center', va='center', fontsize=8, style='italic', color='#6b7280')
    
    # Add legend
    legend_y = 0.8
    ax.text(1, legend_y, "Legend:", fontsize=9, weight='bold')
    ax.text(1, legend_y - 0.4, "⭕ Circle  ▭ Rectangle  │ X-Axis  ─ Y-Axis  ✂️ Splitline", 
            fontsize=8, color='#4b5563')
    
    # Title
    ax.text(7, 11.5, f"PBC Decision Tree ({len(rules)} Rules, {n_classes} Classes)", 
            ha='center', va='center', fontsize=13, weight='bold', color='#1f2937')
    
    return fig

# ==================== SESSION STATE INITIALIZATION ====================

if 'rules' not in st.session_state:
    st.session_state.rules = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'pbc_classifier' not in st.session_state:
    st.session_state.pbc_classifier = PBCClassifier()
if 'tree_nodes' not in st.session_state:
    st.session_state.tree_nodes = []
if 'current_node_id' not in st.session_state:
    st.session_state.current_node_id = None
if 'splitline_selected' not in st.session_state:
    st.session_state.splitline_selected = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'split_screen'

# ==================== APPLY THEME ====================

apply_theme(st.session_state.dark_mode)

# ==================== HEADER ====================

col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    st.title("🌳 PBC Visual Classification Dashboard (Hybrid)")
    st.markdown("*Perception-Based Classification: Circle Segments + Scatter Plot Approach*")
with col2:
    view_mode = st.selectbox("View", ["Split Screen", "Tabbed"], label_visibility="collapsed")
    st.session_state.view_mode = view_mode.lower().replace(" ", "_")
with col3:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dataset selection
    st.subheader("📊 Dataset")
    dataset_option = st.selectbox(
        "Select Dataset",
        ["Iris", "Wine", "Breast Cancer", "Upload CSV"]
    )
    
    # Load dataset
    if dataset_option == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['class'] = data.target_names[data.target]
    elif dataset_option == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['class'] = data.target_names[data.target]
    elif dataset_option == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['class'] = data.target_names[data.target]
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'class' not in df.columns:
                st.error("CSV must have a 'class' column")
                st.stop()
        else:
            st.info("Please upload a CSV file")
            st.stop()
    
    st.session_state.dataset = df
    
    # Initialize root node if empty
    if not st.session_state.tree_nodes:
        root = TreeNode(0, list(range(len(df))), depth=0)
        st.session_state.tree_nodes.append(root)
        st.session_state.current_node_id = 0
    
    # Feature selection
    st.subheader("🎯 Features for Scatter Plot")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    x_feature = st.selectbox("X-Axis Feature", numeric_cols, index=0, key='x_feat')
    y_feature = st.selectbox("Y-Axis Feature", numeric_cols, 
                            index=1 if len(numeric_cols) > 1 else 0, key='y_feat')
    
    # Circle Segments features
    st.subheader("🎯 Circle Segments Features")
    max_features = min(12, len(numeric_cols))
    circle_features = st.multiselect(
        "Select Attributes (4-12 recommended)",
        numeric_cols,
        default=numeric_cols[:min(6, len(numeric_cols))],
        max_selections=12
    )
    
    st.divider()
    
    # Current node info
    st.subheader("📍 Current Node")
    current_node = next((n for n in st.session_state.tree_nodes 
                        if n.node_id == st.session_state.current_node_id), None)
    
    if current_node:
        st.info(f"""
        **Node ID**: {current_node.node_id}  
        **Samples**: {len(current_node.data_indices)}  
        **Depth**: {current_node.depth}  
        **Status**: {'🍃 Leaf' if current_node.is_leaf else '🌿 Internal'}
        """)
        
        if current_node.used_attributes:
            st.write("**Used Attributes:**")
            for attr in current_node.used_attributes:
                st.write(f"  • {attr}")
    
    st.divider()
    
    # Rule creation
    st.subheader("✏️ Create Rules")
    
    rule_method = st.radio(
        "Rule Creation Method",
        ["Scatter Plot Rules", "Circle Segments Splitline"],
        help="Choose how to create classification rules"
    )
    
    predicted_class = st.selectbox(
        "Assign Class",
        df['class'].unique(),
        key='pred_class'
    )
    
    if rule_method == "Scatter Plot Rules":
        rule_type = st.selectbox(
            "Rule Type",
            ["Circle", "Rectangle", "X-Axis Threshold", "Y-Axis Threshold"],
            key='rule_type'
        )
        
        if rule_type == "Circle":
            st.write("**Circle Parameters**")
            cx = st.number_input("Center X", value=float(df[x_feature].mean()), key='cx')
            cy = st.number_input("Center Y", value=float(df[y_feature].mean()), key='cy')
            r = st.number_input("Radius", value=1.0, min_value=0.1, key='r')
            
            if st.button("➕ Add Circle Rule", key='add_circle'):
                rule = Rule('circle', {'cx': cx, 'cy': cy, 'r': r}, predicted_class, 
                           st.session_state.current_node_id)
                st.session_state.pbc_classifier.add_rule(rule)
                st.success("Circle rule added!")
                st.rerun()
        
        elif rule_type == "Rectangle":
            st.write("**Rectangle Parameters**")
            x1 = st.number_input("X Min", value=float(df[x_feature].min()), key='x1')
            x2 = st.number_input("X Max", value=float(df[x_feature].max()), key='x2')
            y1 = st.number_input("Y Min", value=float(df[y_feature].min()), key='y1')
            y2 = st.number_input("Y Max", value=float(df[y_feature].max()), key='y2')
            
            if st.button("➕ Add Rectangle Rule", key='add_rect'):
                rule = Rule('rectangle', {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, 
                           predicted_class, st.session_state.current_node_id)
                st.session_state.pbc_classifier.add_rule(rule)
                st.success("Rectangle rule added!")
                st.rerun()
        
        elif rule_type == "X-Axis Threshold":
            st.write("**X-Axis Threshold**")
            threshold = st.number_input("Threshold Value", value=float(df[x_feature].median()), key='x_thresh')
            direction = st.radio("Apply to", ["Left (≤)", "Right (>)"], key='x_dir')
            
            if st.button("➕ Add X-Axis Rule", key='add_x'):
                rule = Rule('axis_x', 
                           {'threshold': threshold, 'direction': 'left' if direction == "Left (≤)" else 'right'}, 
                           predicted_class, st.session_state.current_node_id)
                st.session_state.pbc_classifier.add_rule(rule)
                st.success("X-Axis rule added!")
                st.rerun()
        
        else:  # Y-Axis Threshold
            st.write("**Y-Axis Threshold**")
            threshold = st.number_input("Threshold Value", value=float(df[y_feature].median()), key='y_thresh')
            direction = st.radio("Apply to", ["Below (≤)", "Above (>)"], key='y_dir')
            
            if st.button("➕ Add Y-Axis Rule", key='add_y'):
                rule = Rule('axis_y', 
                           {'threshold': threshold, 'direction': 'below' if direction == "Below (≤)" else 'above'}, 
                           predicted_class, st.session_state.current_node_id)
                st.session_state.pbc_classifier.add_rule(rule)
                st.success("Y-Axis rule added!")
                st.rerun()
    
    else:  # Circle Segments Splitline
        st.write("**Splitline Selection**")
        
        if circle_features:
            split_attribute = st.selectbox(
                "Select Attribute",
                circle_features,
                key='split_attr'
            )
            
            # Get current node data
            if current_node:
                node_data = df.iloc[current_node.data_indices]
                attr_values = node_data[split_attribute].sort_values()
                
                split_value = st.slider(
                    f"Split Point for {split_attribute}",
                    float(attr_values.min()),
                    float(attr_values.max()),
                    float(attr_values.median()),
                    key='split_val'
                )
                
                direction = st.radio(
                    "Assign class to",
                    ["Left (≤ split)", "Right (> split)"],
                    key='split_dir'
                )
                
                # Show magnified view
                with st.expander("🔍 Magnified Splitline View"):
                    # Show values around split point
                    close_values = attr_values[
                        (attr_values >= split_value - 0.5) & 
                        (attr_values <= split_value + 0.5)
                    ]
                    
                    fig_mag = go.Figure()
                    for val in close_values:
                        class_label = node_data[node_data[split_attribute] == val]['class'].iloc[0]
                        color = 'red' if val <= split_value else 'blue'
                        
                        fig_mag.add_trace(go.Scatter(
                            x=[val],
                            y=[0],
                            mode='markers',
                            marker=dict(size=15, color=color),
                            name=class_label,
                            showlegend=False,
                            hovertext=f"Value: {val:.2f}, Class: {class_label}"
                        ))
                    
                    fig_mag.add_vline(x=split_value, line_dash="dash", 
                                     line_color="green", line_width=3,
                                     annotation_text="Split Point")
                    
                    fig_mag.update_layout(
                        title="Magnified View Around Split Point",
                        xaxis_title=split_attribute,
                        height=200,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_mag, use_container_width=True)
                
                if st.button("✂️ Add Splitline Rule", key='add_split'):
                    rule = Rule('splitline', 
                               {'attribute': split_attribute, 
                                'value': split_value,
                                'direction': 'left' if direction == "Left (≤ split)" else 'right'}, 
                               predicted_class, st.session_state.current_node_id)
                    st.session_state.pbc_classifier.add_rule(rule)
                    
                    # Update tree structure
                    if current_node:
                        current_node.split_attribute = split_attribute
                        current_node.split_points.append(split_value)
                        current_node.used_attributes.add(split_attribute)
                    
                    st.success(f"Splitline rule added: {split_attribute} at {split_value:.2f}")
                    st.rerun()
        else:
            st.warning("Please select circle segment features first")
    
    st.divider()
    
    # Tree operations
    st.subheader("🌳 Tree Operations")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ Undo Last", use_container_width=True):
            if st.session_state.pbc_classifier.undo_last_action():
                st.success("Undone!")
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.pbc_classifier = PBCClassifier()
            st.session_state.tree_nodes = [TreeNode(0, list(range(len(df))), depth=0)]
            st.session_state.current_node_id = 0
            st.rerun()
    
    if current_node and not current_node.is_leaf:
        if st.button("🍃 Mark as Leaf", use_container_width=True):
            current_node.is_leaf = True
            current_node.class_label = predicted_class
            st.success(f"Node {current_node.node_id} marked as leaf: {predicted_class}")
            st.rerun()
    
    # Rules management with edit/delete
    st.subheader("📋 Active Rules")
    if st.session_state.pbc_classifier.rules:
        st.write(f"Total Rules: {len(st.session_state.pbc_classifier.rules)}")
        
        for idx, rule in enumerate(st.session_state.pbc_classifier.rules):
            with st.expander(f"Rule {idx+1}: {rule.rule_type.upper()} → {rule.predicted_class}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.json(rule.params)
                    
                    # Edit functionality
                    new_class = st.selectbox(
                        "Change Class",
                        df['class'].unique(),
                        index=list(df['class'].unique()).index(rule.predicted_class),
                        key=f'edit_class_{idx}'
                    )
                    
                    if st.button(f"💾 Update Class", key=f'update_{idx}'):
                        rule.predicted_class = new_class
                        st.success(f"Rule {idx+1} updated!")
                        st.rerun()
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f'delete_{idx}', type='secondary'):
                        st.session_state.pbc_classifier.rules.pop(idx)
                        st.success(f"Rule {idx+1} deleted!")
                        st.rerun()
    else:
        st.info("No rules created yet")

# ==================== MAIN CONTENT ====================

# Define class colors
classes = df['class'].unique()
# Use matplotlib-compatible hex colors
class_colors = {
    classes[0]: '#e41a1c',  # Red
    classes[1]: '#377eb8',  # Blue
    classes[2]: '#4daf4a',  # Green
}
# Add more colors if needed
additional_colors = ['#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
for i, cls in enumerate(classes[3:], start=3):
    class_colors[cls] = additional_colors[min(i-3, len(additional_colors)-1)]

# Get current node data
current_node = next((n for n in st.session_state.tree_nodes 
                    if n.node_id == st.session_state.current_node_id), None)

if current_node:
    current_df = df.iloc[current_node.data_indices].copy()
else:
    current_df = df.copy()

# Available features (excluding used ones)
available_circle_features = [f for f in circle_features 
                            if current_node and f not in current_node.used_attributes]

# ==================== SPLIT SCREEN VS TABBED ====================

if st.session_state.view_mode == 'split_screen':
    # Split Screen Layout (TRUE PBC STYLE)
    st.markdown("### 📺 Split Screen View - PBC Method")
    
    col_data, col_knowledge = st.columns([1, 1])
    
    with col_data:
        st.markdown("#### 📊 Data Interaction Window")
        
        # Visualization mode selector
        viz_mode = st.radio(
            "Visualization",
            ["Circle Segments", "Scatter Plot"],
            horizontal=True,
            key='viz_mode_split'
        )
        
        if viz_mode == "Circle Segments":
            if available_circle_features:
                fig_circle = create_circle_segments_plot(
                    current_df, 
                    available_circle_features, 
                    classes, 
                    class_colors
                )
                st.plotly_chart(fig_circle, use_container_width=True)
                
                st.caption(f"Showing {len(current_df)} samples from Node {current_node.node_id if current_node else 0}")
                st.caption(f"Available attributes: {len(available_circle_features)}")
            else:
                st.warning("All attributes have been used in this path. Mark as leaf or backtrack.")
        
        else:  # Scatter Plot
            fig = go.Figure()
            
            # Plot data points
            for cls in classes:
                mask = current_df['class'] == cls
                fig.add_trace(go.Scatter(
                    x=current_df[mask][x_feature],
                    y=current_df[mask][y_feature],
                    mode='markers',
                    name=cls,
                    marker=dict(size=8, opacity=0.7, color=class_colors.get(cls))
                ))
            
            # Overlay rules
            for idx, rule in enumerate(st.session_state.pbc_classifier.rules):
                if rule.rule_type == 'circle':
                    theta = np.linspace(0, 2*np.pi, 100)
                    x_circle = rule.params['cx'] + rule.params['r'] * np.cos(theta)
                    y_circle = rule.params['cy'] + rule.params['r'] * np.sin(theta)
                    fig.add_trace(go.Scatter(
                        x=x_circle, y=y_circle,
                        mode='lines',
                        name=f'Rule {idx+1}: {rule.predicted_class}',
                        line=dict(dash='dash', width=2),
                        showlegend=False
                    ))
                elif rule.rule_type == 'rectangle':
                    fig.add_shape(
                        type="rect",
                        x0=rule.params['x1'], y0=rule.params['y1'],
                        x1=rule.params['x2'], y1=rule.params['y2'],
                        line=dict(dash='dash', width=2),
                    )
                elif rule.rule_type == 'axis_x':
                    fig.add_vline(
                        x=rule.params['threshold'],
                        line_dash="dash",
                        annotation_text=f"X: {rule.predicted_class}",
                        line_width=2
                    )
                elif rule.rule_type == 'axis_y':
                    fig.add_hline(
                        y=rule.params['threshold'],
                        line_dash="dash",
                        annotation_text=f"Y: {rule.predicted_class}",
                        line_width=2
                    )
            
            fig.update_layout(
                title=f"Scatter Plot: {x_feature} vs {y_feature}",
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                height=600,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Showing {len(current_df)} samples from Node {current_node.node_id if current_node else 0}")
    
        with col2:
            st.markdown("#### 🌳 Knowledge Interaction Window")
            
            # Tree visualization
            if st.session_state.pbc_classifier.rules:
                fig_tree = create_pbc_tree_visualization(
                    st.session_state.pbc_classifier.rules,
                    classes,
                    class_colors
                )
                if fig_tree:
                    st.pyplot(fig_tree)
            else:
                st.info("No rules yet. Create your first rule to start building the tree!")
            
            # Node statistics
        st.markdown("##### 📊 Current Node Statistics")
        if current_node:
            class_dist = current_df['class'].value_counts()
            
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=class_dist.index,
                    y=class_dist.values,
                    marker_color=[class_colors.get(cls) for cls in class_dist.index],
                    text=class_dist.values,
                    textposition='auto'
                )
            ])
            fig_dist.update_layout(
                title=f"Class Distribution in Node {current_node.node_id}",
                xaxis_title="Class",
                yaxis_title="Count",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Purity metrics
            total = len(current_df)
            if total > 0:
                majority_class = class_dist.index[0]
                majority_count = class_dist.values[0]
                purity = (majority_count / total) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Samples", total)
                col2.metric("Purity", f"{purity:.1f}%")
                col3.metric("Classes", len(class_dist))
                
                if purity >= 90:
                    st.success(f"✨ High purity! Consider marking as leaf: {majority_class}")
                elif purity >= 70:
                    st.info(f"Good purity. Dominant class: {majority_class}")
                else:
                    st.warning(f"Low purity. More splits needed.")

else:  # Tabbed view
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Scatter Plot View", 
        "🎯 Circle Segments View", 
        "📈 Evaluation", 
        "🌳 Decision Tree"
    ])
    
    with tab1:
        st.header("Interactive Scatter Plot Visualization")
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot data points
        for cls in classes:
            mask = current_df['class'] == cls
            fig.add_trace(go.Scatter(
                x=current_df[mask][x_feature],
                y=current_df[mask][y_feature],
                mode='markers',
                name=cls,
                marker=dict(size=8, opacity=0.7, color=class_colors.get(cls))
            ))
        
        # Overlay rules
        for idx, rule in enumerate(st.session_state.pbc_classifier.rules):
            if rule.rule_type == 'circle':
                theta = np.linspace(0, 2*np.pi, 100)
                x_circle = rule.params['cx'] + rule.params['r'] * np.cos(theta)
                y_circle = rule.params['cy'] + rule.params['r'] * np.sin(theta)
                fig.add_trace(go.Scatter(
                    x=x_circle, y=y_circle,
                    mode='lines',
                    name=f'Rule {idx+1}: {rule.predicted_class}',
                    line=dict(dash='dash', width=2)
                ))
            elif rule.rule_type == 'rectangle':
                fig.add_shape(
                    type="rect",
                    x0=rule.params['x1'], y0=rule.params['y1'],
                    x1=rule.params['x2'], y1=rule.params['y2'],
                    line=dict(dash='dash', width=2),
                )
            elif rule.rule_type == 'axis_x':
                fig.add_vline(
                    x=rule.params['threshold'],
                    line_dash="dash",
                    annotation_text=f"X: {rule.predicted_class}",
                    line_width=2
                )
            elif rule.rule_type == 'axis_y':
                fig.add_hline(
                    y=rule.params['threshold'],
                    line_dash="dash",
                    annotation_text=f"Y: {rule.predicted_class}",
                    line_width=2
                )
        
        fig.update_layout(
            title=f"Feature Space: {x_feature} vs {y_feature}",
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display rule summary
        if st.session_state.pbc_classifier.rules:
            st.subheader("📝 Rule Summary")
            for idx, rule in enumerate(st.session_state.pbc_classifier.rules):
                with st.expander(f"Rule {idx+1}: {rule.rule_type.upper()} → {rule.predicted_class}"):
                    st.json(rule.params)
    
    with tab2:
        st.header("🎯 Circle Segments Visualization (PBC Paper Method)")
        st.markdown("""
        This implements the **Circle Segments** technique from the original PBC paper. 
        Each segment represents one attribute, with data points arranged radially and sorted by attribute value.
        Colors represent class labels.
        """)
        
        if available_circle_features:
            # Splitting strategy guidance
            st.info("""
            **📚 PBC Splitting Strategy (from paper):**
            1. **BPP** (Best Pure Partitions): Choose segment with largest pure regions
            2. **LCP** (Largest Cluster Partitioning): Select segment with largest homogeneous cluster
            3. **BCP** (Best Complete Partitioning): Pick segment with best separable partitions
            4. **DDP** (Different Distribution Partitioning): Use when distributions differ clearly
            """)
            
            fig_circle = create_circle_segments_plot(
                current_df, 
                available_circle_features, 
                classes, 
                class_colors
            )
            st.plotly_chart(fig_circle, use_container_width=True)
            
            st.caption(f"📊 Displaying {len(current_df)} samples from Node {current_node.node_id if current_node else 0}")
            st.caption(f"🎯 Available attributes: {', '.join(available_circle_features)}")
            
            # Interactive analysis
            st.subheader("🔍 Attribute Analysis")
            
            analysis_attr = st.selectbox("Analyze Attribute", available_circle_features)
            
            if analysis_attr:
                attr_data = current_df[[analysis_attr, 'class']].copy()
                attr_data = attr_data.sort_values(by=analysis_attr)
                
                # Create detailed view
                fig_detail = go.Figure()
                
                for cls in classes:
                    mask = attr_data['class'] == cls
                    fig_detail.add_trace(go.Scatter(
                        x=attr_data[mask].index,
                        y=attr_data[mask][analysis_attr],
                        mode='markers',
                        name=cls,
                        marker=dict(size=8, color=class_colors.get(cls))
                    ))
                
                fig_detail.update_layout(
                    title=f"Sorted View: {analysis_attr}",
                    xaxis_title="Sorted Index",
                    yaxis_title=analysis_attr,
                    height=300
                )
                
                st.plotly_chart(fig_detail, use_container_width=True)
                
                # Suggest split points
                st.write("**💡 Suggested Split Points:**")
                
                for cls in classes:
                    cls_data = attr_data[attr_data['class'] == cls][analysis_attr]
                    if len(cls_data) > 0:
                        st.write(f"- **{cls}**: Range [{cls_data.min():.2f}, {cls_data.max():.2f}]")
        else:
            st.warning("⚠️ All attributes have been used in this path. Consider marking as leaf or backtracking.")
    
    with tab3:
        st.header("📈 Model Evaluation & Comparison")
        
        if not st.session_state.pbc_classifier.rules:
            st.warning("⚠️ Please create at least one rule to evaluate the PBC model")
        else:
            # Prepare data
            X = df[[x_feature, y_feature]].values
            y_true = df['class'].values
            
            # PBC predictions
            default_class = df['class'].mode()[0]
            y_pred_pbc = st.session_state.pbc_classifier.predict(X[:, 0], X[:, 1], default_class=default_class)
            
            # Train sklearn DecisionTree for comparison
            dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
            dt_classifier.fit(X, y_true)
            y_pred_dt = dt_classifier.predict(X)
            
            # Calculate metrics
            acc_pbc = accuracy_score(y_true, y_pred_pbc)
            acc_dt = accuracy_score(y_true, y_pred_dt)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎨 PBC Model (Your Rules)</h3>
                    <h1>{acc_pbc*100:.1f}%</h1>
                    <p>Accuracy</p>
                    <p>Rules: {len(st.session_state.pbc_classifier.rules)}</p>
                    <p>Tree Nodes: {len(st.session_state.tree_nodes)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("Confusion Matrix - PBC")
                cm_pbc = confusion_matrix(y_true, y_pred_pbc, labels=classes)
                fig_cm_pbc = px.imshow(
                    cm_pbc,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=classes,
                    y=classes,
                    text_auto=True,
                    color_continuous_scale='Blues'
                )
                fig_cm_pbc.update_layout(height=400)
                st.plotly_chart(fig_cm_pbc, use_container_width=True)
                
                st.text("Classification Report - PBC")
                report_pbc = classification_report(y_true, y_pred_pbc)
                st.text(report_pbc)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌳 Decision Tree (sklearn)</h3>
                    <h1>{acc_dt*100:.1f}%</h1>
                    <p>Accuracy</p>
                    <p>Depth: {dt_classifier.get_depth()}</p>
                    <p>Nodes: {dt_classifier.tree_.node_count}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("Confusion Matrix - Decision Tree")
                cm_dt = confusion_matrix(y_true, y_pred_dt, labels=classes)
                fig_cm_dt = px.imshow(
                    cm_dt,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=classes,
                    y=classes,
                    text_auto=True,
                    color_continuous_scale='Greens'
                )
                fig_cm_dt.update_layout(height=400)
                st.plotly_chart(fig_cm_dt, use_container_width=True)
                
                st.text("Classification Report - Decision Tree")
                report_dt = classification_report(y_true, y_pred_dt)
                st.text(report_dt)
            
            # Comparison insights
            st.subheader("🔍 Model Comparison Insights")
            
            diff = acc_pbc - acc_dt
            if abs(diff) < 0.01:
                st.success("✨ Both models perform equally well!")
            elif diff > 0:
                st.success(f"🎉 PBC model outperforms Decision Tree by {diff*100:.1f}%!")
            else:
                st.info(f"Decision Tree performs better by {abs(diff)*100:.1f}%. Try adding more rules or adjusting existing ones!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Interpretability Winner", 
                         "PBC" if len(st.session_state.pbc_classifier.rules) < dt_classifier.tree_.node_count else "Decision Tree",
                         f"{len(st.session_state.pbc_classifier.rules)} vs {dt_classifier.tree_.node_count} nodes")
            
            with col2:
                st.metric("Accuracy Winner",
                         "PBC" if acc_pbc >= acc_dt else "Decision Tree",
                         f"{max(acc_pbc, acc_dt)*100:.1f}%")
            
            with col3:
                flexibility = len(st.session_state.tree_nodes) - len([n for n in st.session_state.tree_nodes if n.is_leaf])
                st.metric("Active Branches", flexibility)
            
            st.markdown("---")
            st.markdown("""
            ### 📊 Key Advantages of PBC:
            - ✅ **Domain Knowledge Integration**: You can use your expertise
            - ✅ **Non-binary Splits**: Multiple split points per attribute
            - ✅ **Backtracking**: Undo and try different approaches
            - ✅ **Visual Understanding**: Deep insight into data patterns
            - ✅ **Interpretability**: Smaller trees, clearer rules
            """)
    
    with tab4:
        st.header("🌳 Decision Tree Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("PBC Tree Structure")
            if st.session_state.pbc_classifier.rules:
                fig_tree = create_pbc_tree_visualization(
                    st.session_state.pbc_classifier.rules,
                    classes,
                    class_colors
                )
                if fig_tree:
                    st.pyplot(fig_tree)
            else:
                st.info("No rules created yet. Start by creating rules!")
        
        with col2:
            st.subheader("Tree Statistics")
            
            total_rules = len(st.session_state.pbc_classifier.rules)
            rules_by_class = {}
            rules_by_type = {}
            
            for rule in st.session_state.pbc_classifier.rules:
                # Count by class
                if rule.predicted_class not in rules_by_class:
                    rules_by_class[rule.predicted_class] = 0
                rules_by_class[rule.predicted_class] += 1
                
                # Count by type
                if rule.rule_type not in rules_by_type:
                    rules_by_type[rule.rule_type] = 0
                rules_by_type[rule.rule_type] += 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rules", total_rules)
            with col2:
                st.metric("Classes", len(rules_by_class))
            
            if rules_by_class:
                st.write("**📊 Rules per Class:**")
                for cls, count in sorted(rules_by_class.items()):
                    percentage = (count / total_rules) * 100
                    st.progress(percentage / 100)
                    st.caption(f"{cls}: {count} rules ({percentage:.1f}%)")
            
            if rules_by_type:
                st.divider()
                st.write("**🎯 Rules by Type:**")
                for rtype, count in sorted(rules_by_type.items()):
                    icon = {'circle': '⭕', 'rectangle': '▭', 'axis_x': '│', 'axis_y': '─', 'splitline': '✂️'}.get(rtype, '●')
                    st.write(f"{icon} {rtype.title()}: {count}")
            
            st.divider()
            
            st.subheader("Rule Details")
            if st.session_state.pbc_classifier.rules:
                for idx, rule in enumerate(st.session_state.pbc_classifier.rules):
                    with st.expander(f"Rule {idx+1}: {rule.rule_type}"):
                        st.write(f"**Type**: {rule.rule_type}")
                        st.write(f"**Class**: {rule.predicted_class}")
                        st.write(f"**Parameters**:")
                        st.json(rule.params)
            else:
                st.info("No rules yet")
        
        # sklearn Decision Tree comparison
        st.divider()
        st.subheader("sklearn Decision Tree (for comparison)")
        
        if st.session_state.pbc_classifier.rules:
            X = df[[x_feature, y_feature]].values
            y = df['class'].values
            
            dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            dt.fit(X, y)
            
            fig_sklearn, ax = plt.subplots(figsize=(15, 10))
            plot_tree(
                dt,
                feature_names=[x_feature, y_feature],
                class_names=classes,
                filled=True,
                ax=ax,
                fontsize=10
            )
            st.pyplot(fig_sklearn)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("sklearn Depth", dt.get_depth())
            col2.metric("sklearn Leaves", dt.get_n_leaves())
            col3.metric("sklearn Nodes", dt.tree_.node_count)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>🌳 PBC Visual Classification Dashboard (Hybrid Implementation)</b></p>
    <p><i>Combining Circle Segments (PBC Paper) + Modern Scatter Plot Approach</i></p>
    <p>Based on: <b>"Visual Classification: An Interactive Approach to Decision Tree Construction"</b></p>
    <p>Ankerst, Elsen, Ester, Kriegel (1999)</p>
</div>
""", unsafe_allow_html=True)