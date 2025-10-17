import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
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

# Custom CSS for themes
def apply_theme(dark_mode):
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
        }
        </style>
        """, unsafe_allow_html=True)

# Rule class for PBC
class Rule:
    def __init__(self, rule_type, params, predicted_class):
        self.rule_type = rule_type  # 'circle', 'rectangle', 'axis_x', 'axis_y'
        self.params = params
        self.predicted_class = predicted_class
    
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
        return False

# PBC Classifier
class PBCClassifier:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule):
        self.rules.append(rule)
    
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

# Initialize session state
if 'rules' not in st.session_state:
    st.session_state.rules = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'circle_segments_splits' not in st.session_state:
    st.session_state.circle_segments_splits = {}

# Apply theme
apply_theme(st.session_state.dark_mode)

# Header
col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    st.title("🌳 PBC Visual Classification Dashboard")
    st.markdown("*Interactive Decision Tree Construction using Perception-Based Classification*")
with col3:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Sidebar
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
    
    # Feature selection
    st.subheader("🎯 Features")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    x_feature = st.selectbox("X-Axis Feature", numeric_cols, index=0)
    y_feature = st.selectbox("Y-Axis Feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    # Visualization mode
    st.subheader("👁️ Visualization Mode")
    viz_mode = st.radio(
        "Select Mode",
        ["Scatter Plot", "Circle Segments"],
        help="Scatter Plot: Traditional 2D view | Circle Segments: PBC paper's radial visualization"
    )
    
    # Rule creation
    st.subheader("✏️ Create Rules")
    rule_type = st.selectbox(
        "Rule Type",
        ["Circle", "Rectangle", "X-Axis Threshold", "Y-Axis Threshold"]
    )
    
    predicted_class = st.selectbox(
        "Assign Class",
        df['class'].unique()
    )
    
    # Rule parameters based on type
    if rule_type == "Circle":
        st.write("**Circle Parameters**")
        cx = st.number_input("Center X", value=float(df[x_feature].mean()))
        cy = st.number_input("Center Y", value=float(df[y_feature].mean()))
        r = st.number_input("Radius", value=1.0, min_value=0.1)
        
        if st.button("➕ Add Circle Rule"):
            rule = Rule('circle', {'cx': cx, 'cy': cy, 'r': r}, predicted_class)
            st.session_state.rules.append(rule)
            st.success("Circle rule added!")
            st.rerun()
    
    elif rule_type == "Rectangle":
        st.write("**Rectangle Parameters**")
        x1 = st.number_input("X Min", value=float(df[x_feature].min()))
        x2 = st.number_input("X Max", value=float(df[x_feature].max()))
        y1 = st.number_input("Y Min", value=float(df[y_feature].min()))
        y2 = st.number_input("Y Max", value=float(df[y_feature].max()))
        
        if st.button("➕ Add Rectangle Rule"):
            rule = Rule('rectangle', {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, predicted_class)
            st.session_state.rules.append(rule)
            st.success("Rectangle rule added!")
            st.rerun()
    
    elif rule_type == "X-Axis Threshold":
        st.write("**X-Axis Threshold**")
        threshold = st.number_input("Threshold Value", value=float(df[x_feature].median()))
        direction = st.radio("Apply to", ["Left (≤)", "Right (>)"])
        
        if st.button("➕ Add X-Axis Rule"):
            rule = Rule('axis_x', 
                       {'threshold': threshold, 'direction': 'left' if direction == "Left (≤)" else 'right'}, 
                       predicted_class)
            st.session_state.rules.append(rule)
            st.success("X-Axis rule added!")
            st.rerun()
    
    else:  # Y-Axis Threshold
        st.write("**Y-Axis Threshold**")
        threshold = st.number_input("Threshold Value", value=float(df[y_feature].median()))
        direction = st.radio("Apply to", ["Below (≤)", "Above (>)"])
        
        if st.button("➕ Add Y-Axis Rule"):
            rule = Rule('axis_y', 
                       {'threshold': threshold, 'direction': 'below' if direction == "Below (≤)" else 'above'}, 
                       predicted_class)
            st.session_state.rules.append(rule)
            st.success("Y-Axis rule added!")
            st.rerun()
    
    # Rules management
    # Rules management
    st.subheader("📋 Active Rules")

    if st.session_state.rules:
        st.write(f"Total Rules: {len(st.session_state.rules)}")

        for idx, rule in enumerate(st.session_state.rules):
            with st.expander(f"Rule {idx+1}: {rule.rule_type.upper()} → {rule.predicted_class}"):
                st.json(rule.params)
                delete_col1, delete_col2 = st.columns([4, 1])
                with delete_col2:
                    if st.button("🗑️"):
                        del st.session_state.rules[idx]
                        st.success(f"Deleted Rule {idx+1}")
                        st.rerun()

        st.markdown("---")
        if st.button("🧹 Clear All Rules"):
            st.session_state.rules = []
            st.rerun()

    else:
        st.info("No rules created yet")


# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🎯 Circle Segments", "📈 Evaluation", "🌳 Decision Tree"])

with tab1:
    st.header("Interactive Feature Space Visualization")
    
    if viz_mode == "Scatter Plot":
        # Create scatter plot
        fig = go.Figure()
        
        # Plot data points
        for cls in df['class'].unique():
            mask = df['class'] == cls
            fig.add_trace(go.Scatter(
                x=df[mask][x_feature],
                y=df[mask][y_feature],
                mode='markers',
                name=cls,
                marker=dict(size=8, opacity=0.7)
            ))
        
        # Overlay rules
        for idx, rule in enumerate(st.session_state.rules):
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
    if st.session_state.rules:
        st.subheader("📝 Rule Summary")
        for idx, rule in enumerate(st.session_state.rules):
            with st.expander(f"Rule {idx+1}: {rule.rule_type.upper()} → {rule.predicted_class}"):
                st.json(rule.params)

with tab2:
    st.header("🎯 Circle Segments Visualization")
    st.markdown("""
    This is the **Circle Segments** technique from the PBC paper. Each segment represents 
    one attribute, with data points arranged radially. Colors represent class labels.
    """)
    
    # Select attributes for circle segments
    selected_features = st.multiselect(
        "Select Attributes for Visualization",
        numeric_cols,
        default=numeric_cols[:min(8, len(numeric_cols))]
    )
    
    if selected_features:
        # Create circle segments visualization
        fig = go.Figure()
        
        n_features = len(selected_features)
        angle_per_feature = 360 / n_features
        
        # Normalize data
        normalized_data = df[selected_features].copy()
        for col in selected_features:
            normalized_data[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Plot each feature segment
        for idx, feature in enumerate(selected_features):
            start_angle = idx * angle_per_feature
            end_angle = (idx + 1) * angle_per_feature
            
            # Sort by feature value
            sorted_indices = normalized_data[feature].argsort()
            
            # Create points in segment
            for i, data_idx in enumerate(sorted_indices):
                r = 0.5 + (i / len(sorted_indices)) * 0.5  # Radius from 0.5 to 1
                angle = start_angle + (end_angle - start_angle) * 0.5
                angle_rad = np.radians(angle)
                
                x = r * np.cos(angle_rad)
                y = r * np.sin(angle_rad)
                
                class_label = df.iloc[data_idx]['class']
                
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(size=3),
                    name=class_label,
                    showlegend=(idx == 0 and i == 0),
                    hovertext=f"{feature}: {df.iloc[data_idx][feature]:.2f}<br>Class: {class_label}"
                ))
            
            # Add segment boundaries
            for angle in [start_angle, end_angle]:
                angle_rad = np.radians(angle)
                fig.add_trace(go.Scatter(
                    x=[0, 1.1*np.cos(angle_rad)],
                    y=[0, 1.1*np.sin(angle_rad)],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))
            
            # Add feature label
            label_angle = (start_angle + end_angle) / 2
            label_r = 1.2
            label_x = label_r * np.cos(np.radians(label_angle))
            label_y = label_r * np.sin(np.radians(label_angle))
            fig.add_annotation(
                x=label_x, y=label_y,
                text=feature,
                showarrow=False,
                font=dict(size=10)
            )
        
        fig.update_layout(
            title="Circle Segments Visualization",
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1.5, 1.5]),
            height=700,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interactive splitline selection
        st.subheader("✂️ Interactive Splitline Selection")
        selected_attr = st.selectbox("Select Attribute for Splitting", selected_features)
        
        if selected_attr:
            sorted_values = df[selected_attr].sort_values()
            split_value = st.slider(
                f"Split Point for {selected_attr}",
                float(sorted_values.min()),
                float(sorted_values.max()),
                float(sorted_values.median())
            )
            
            split_class = st.selectbox("Assign Class to Split", df['class'].unique())
            
            if st.button("Add Split Rule"):
                # This creates an axis threshold rule based on the selected attribute
                st.info(f"Split created: {selected_attr} ≤ {split_value:.2f} → {split_class}")
                st.session_state.circle_segments_splits[selected_attr] = {
                    'value': split_value,
                    'class': split_class
                }
                st.success("Split rule added!")

with tab3:
    st.header("📈 Model Evaluation & Comparison")
    
    if not st.session_state.rules:
        st.warning("⚠️ Please create at least one rule to evaluate the PBC model")
    else:
        # Prepare data
        X = df[[x_feature, y_feature]].values
        y_true = df['class'].values
        
        # PBC predictions
        pbc_classifier = PBCClassifier()
        for rule in st.session_state.rules:
            pbc_classifier.add_rule(rule)
        
        default_class = df['class'].mode()[0]
        y_pred_pbc = pbc_classifier.predict(X[:, 0], X[:, 1], default_class=default_class)
        
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
                <h3>🎨 PBC Model</h3>
                <h1>{acc_pbc*100:.1f}%</h1>
                <p>Accuracy</p>
                <p>Rules: {len(st.session_state.rules)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Confusion Matrix - PBC")
            cm_pbc = confusion_matrix(y_true, y_pred_pbc, labels=df['class'].unique())
            fig_cm_pbc = px.imshow(
                cm_pbc,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=df['class'].unique(),
                y=df['class'].unique(),
                text_auto=True,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm_pbc, use_container_width=True)
            
            st.text("Classification Report - PBC")
            report_pbc = classification_report(y_true, y_pred_pbc)
            st.text(report_pbc)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌳 Decision Tree</h3>
                <h1>{acc_dt*100:.1f}%</h1>
                <p>Accuracy</p>
                <p>Nodes: {dt_classifier.tree_.node_count}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Confusion Matrix - Decision Tree")
            cm_dt = confusion_matrix(y_true, y_pred_dt, labels=df['class'].unique())
            fig_cm_dt = px.imshow(
                cm_dt,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=df['class'].unique(),
                y=df['class'].unique(),
                text_auto=True,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_cm_dt, use_container_width=True)
            
            st.text("Classification Report - Decision Tree")
            report_dt = classification_report(y_true, y_pred_dt)
            st.text(report_dt)
        
        # Comparison insights
        st.subheader("🔍 Model Comparison Insights")
        
        if acc_pbc > acc_dt:
            st.success(f"✨ PBC model outperforms Decision Tree by {(acc_pbc - acc_dt)*100:.1f}%!")
        elif acc_pbc < acc_dt:
            st.info(f"Decision Tree performs better by {(acc_dt - acc_pbc)*100:.1f}%. Try adding more rules!")
        else:
            st.success("Both models perform equally well!")
        
        st.write(f"""
        - **Interpretability**: PBC uses {len(st.session_state.rules)} visual rules vs {dt_classifier.tree_.node_count} nodes
        - **Human-in-the-loop**: PBC incorporates domain knowledge
        - **Flexibility**: PBC allows non-binary splits and backtracking
        """)

with tab4:
    st.header("🌳 Decision Tree Visualization")
    
    if st.session_state.rules:
        # Train decision tree
        X = df[[x_feature, y_feature]].values
        y = df['class'].values
        
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X, y)
        
        # Plot tree
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(
            dt,
            feature_names=[x_feature, y_feature],
            class_names=df['class'].unique(),
            filled=True,
            ax=ax,
            fontsize=10
        )
        st.pyplot(fig)
        
        # Tree statistics
        st.subheader("Tree Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tree Depth", dt.get_depth())
        col2.metric("Number of Leaves", dt.get_n_leaves())
        col3.metric("Total Nodes", dt.tree_.node_count)
    else:
        st.info("Create rules in the sidebar to generate comparisons")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>PBC Visual Classification Dashboard</b> | Inspired by Ankerst et al. (1999)</p>
    <p><i>"Visual Classification: An Interactive Approach to Decision Tree Construction"</i></p>
</div>
""", unsafe_allow_html=True)