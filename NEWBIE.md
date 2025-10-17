# 🌳 PBC Visual Classification Dashboard - Newbie Guide

Welcome to the **PBC (Perception-Based Classification) Visual Classification Dashboard**! This guide will help you understand and use the application, even if you're completely new to machine learning or data visualization.

---

## 📚 Table of Contents

1. [What is This Application?](#what-is-this-application)
2. [Quick Start Guide](#quick-start-guide)
3. [Understanding the Interface](#understanding-the-interface)
4. [Step-by-Step Tutorial](#step-by-step-tutorial)
5. [Understanding Rules](#understanding-rules)
6. [Reading the Results](#reading-the-results)
7. [Tips & Best Practices](#tips--best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Features](#advanced-features)
10. [FAQs](#faqs)

---

## 🎯 What is This Application?

### The Big Picture

Imagine you have a dataset of flowers, patients, or customers, and you want to **classify** them into groups (like "Iris Setosa" or "Healthy/Sick"). Instead of letting a computer algorithm do all the work automatically, this application lets **YOU** draw decision boundaries and create classification rules by looking at the data visually.

### Why is This Cool?

- 🧠 **You're in control**: Use your intuition and domain knowledge
- 👁️ **Visual learning**: See patterns in your data immediately
- 🎨 **Interactive**: Draw boundaries with circles, rectangles, and lines
- 📊 **Compare**: See how your manual rules stack up against automatic algorithms

### Based on Research

This app implements the **PBC (Perception-Based Classification)** method from the 1999 paper *"Visual Classification: An Interactive Approach to Decision Tree Construction"* by Mihael Ankerst and colleagues.

---

## 🚀 Quick Start Guide

### Installation

```bash
# Install required packages
pip install streamlit pandas numpy plotly scikit-learn matplotlib

# Run the application
streamlit run pbc_dashboard.py
```

### First Steps (2 Minutes)

1. **Open the app** in your browser (Streamlit will auto-open it)
2. **Select a dataset** from the sidebar (try "Iris" first - it's simple!)
3. **Look at the scatter plot** - see the colorful dots? Each color is a different class
4. **Create your first rule**: 
   - Choose "Circle" rule type
   - Click "Add Circle Rule"
5. **Go to "Evaluation"** tab to see your accuracy!

---

## 🖥️ Understanding the Interface

### Main Layout

```
┌─────────────────────────────────────────────────────────┐
│  🌳 PBC Visual Classification Dashboard          🌙     │
├──────────────┬──────────────────────────────────────────┤
│              │  📊 Visualization                        │
│   SIDEBAR    │  🎯 Circle Segments                      │
│              │  📈 Evaluation                           │
│   Controls   │  🌳 Decision Tree                        │
│   & Settings │                                          │
│              │        [Your data visualized here]       │
│              │                                          │
└──────────────┴──────────────────────────────────────────┘
```

### Sidebar Components

#### 📊 Dataset Section
- **Purpose**: Choose what data you want to work with
- **Options**: 
  - `Iris` - 150 flower samples, 3 species (easiest!)
  - `Wine` - 178 wine samples, 3 types
  - `Breast Cancer` - 569 samples, 2 diagnoses
  - `Upload CSV` - Use your own data

#### 🎯 Features Section
- **Purpose**: Choose which two attributes to visualize
- **X-Axis Feature**: Horizontal dimension
- **Y-Axis Feature**: Vertical dimension
- **Example**: "sepal length" vs "sepal width" for Iris dataset

#### 👁️ Visualization Mode
- **Scatter Plot**: Traditional 2D view (start here!)
- **Circle Segments**: Advanced radial view from the research paper

#### ✏️ Create Rules
This is where the magic happens! You define classification rules.

---

## 📖 Step-by-Step Tutorial

### Tutorial 1: Your First Classification (Iris Dataset)

#### Step 1: Load the Data
```
1. Sidebar → Dataset: Select "Iris"
2. You'll see 150 flower samples automatically loaded
```

#### Step 2: Choose Features
```
1. X-Axis Feature: "sepal length (cm)"
2. Y-Axis Feature: "sepal width (cm)"
3. Look at the scatter plot - see 3 color clusters?
```

#### Step 3: Create Your First Rule
```
1. Rule Type: Select "Circle"
2. Assign Class: "setosa" (usually the red cluster)
3. Center X: Look at the scatter plot, estimate center (around 5.0)
4. Center Y: Estimate center (around 3.5)
5. Radius: 1.0 (you can adjust later)
6. Click "➕ Add Circle Rule"
```

**What just happened?**
You told the system: "Everything inside this circle is probably Setosa!"

#### Step 4: Check Your Work
```
1. Go to "📈 Evaluation" tab
2. Look at your accuracy - probably 50-70% with one rule
3. See which samples you classified correctly (confusion matrix)
```

#### Step 5: Add More Rules
```
1. Go back to "📊 Visualization" tab
2. Create rules for the other two classes
3. Try different rule types (Rectangle, Axis Threshold)
4. Watch your accuracy improve!
```

### Tutorial 2: Understanding Circle Segments

#### What Are Circle Segments?

Imagine a pizza divided into slices. Each slice represents one feature (attribute) of your data. The colors show different classes.

```
        Attribute 3
             │
    Attr 2  │  Attr 4
         \  │  /
          \ │ /
    ───────O───────  Attribute 1
          / │ \
         /  │  \
    Attr 8  │  Attr 5
             │
        Attribute 7
```

#### How to Use It

```
1. Visualization Mode: Select "Circle Segments"
2. Select Attributes: Choose 4-8 features
3. Observe: Look for pure color regions (one class dominates)
4. Create Split: 
   - Select an attribute
   - Move the slider to find a good split point
   - Assign a class
   - Click "Add Split Rule"
```

**Pro Tip**: Look for segments where colors are clearly separated!

---

## 🎨 Understanding Rules

### Rule Types Explained

#### 1️⃣ Circle Rule
**Use when**: You see a round cluster of one class

```
Visual:           What it means:
   ●●●            "If a point is inside this circle,
  ●●●●●           it belongs to Class A"
   ●●●

Example: Center=(5, 3.5), Radius=1.0 → Setosa
```

**Parameters**:
- `Center X`: Horizontal position of circle center
- `Center Y`: Vertical position of circle center
- `Radius`: How big the circle is

#### 2️⃣ Rectangle Rule
**Use when**: You see a box-shaped region of one class

```
Visual:           What it means:
┌─────────┐       "If X is between x1 and x2,
│ ●●●●●●● │       AND Y is between y1 and y2,
│ ●●●●●●● │       it's Class B"
└─────────┘

Example: X: 5.5-7.0, Y: 2.0-3.5 → Versicolor
```

**Parameters**:
- `X Min` and `X Max`: Horizontal boundaries
- `Y Min` and `Y Max`: Vertical boundaries

#### 3️⃣ X-Axis Threshold
**Use when**: You can split classes with a vertical line

```
Visual:           What it means:
    │ ●●●          "If X ≤ threshold, it's Class A"
    │  ●●          OR
●●●●│              "If X > threshold, it's Class B"
 ●●●│

Example: X ≤ 5.5 → Setosa
```

**Parameters**:
- `Threshold`: Where to draw the line
- `Direction`: Left (≤) or Right (>)

#### 4️⃣ Y-Axis Threshold
**Use when**: You can split classes with a horizontal line

```
Visual:           What it means:
●●●●●●●●          "If Y ≤ threshold, it's Class A"
─────────         OR
 ●●●●●●           "If Y > threshold, it's Class B"

Example: Y ≤ 3.0 → Virginica
```

**Parameters**:
- `Threshold`: Where to draw the line
- `Direction`: Below (≤) or Above (>)

---

## 📊 Reading the Results

### The Evaluation Tab

#### 1. Accuracy Scores

```
🎨 PBC Model          🌳 Decision Tree
   85.3%                  86.3%
   
Rules: 4              Nodes: 15
```

**What does this mean?**
- **85.3%**: Your manual rules classified 85.3% of samples correctly
- **86.3%**: The automatic algorithm got 86.3% correct
- **Interpretation**: You're doing great! Almost as good as the algorithm!

#### 2. Confusion Matrix

```
                Predicted
              Set  Ver  Vir
Actual  Set [ 48   2    0 ]
        Ver [  0  45    5 ]
        Vir [  0   3   47 ]
```

**How to read this**:
- **Diagonal (48, 45, 47)**: Correct predictions ✅
- **Off-diagonal (2, 5, 3)**: Mistakes ❌
- **Example**: "2 Setosa were wrongly predicted as Versicolor"

#### 3. Classification Report

```
              precision  recall  f1-score
setosa           1.00     0.96     0.98
versicolor       0.90     0.90     0.90
virginica        0.90     0.94     0.92
```

**Simple explanation**:
- **Precision**: "When I say it's Class A, how often am I right?"
- **Recall**: "Out of all actual Class A samples, how many did I find?"
- **F1-Score**: Average of precision and recall (higher is better)
- **Goal**: Get all scores close to 1.00!

---

## 💡 Tips & Best Practices

### Starting Out

1. **Start Simple**
   - Use Iris dataset first
   - Create 1-2 rules only
   - Understand what each rule does

2. **Visual First**
   - Look at the scatter plot carefully
   - Identify obvious clusters
   - Draw rules around pure color regions

3. **Iterate**
   - Add one rule at a time
   - Check accuracy after each rule
   - Adjust if accuracy goes down

### Creating Good Rules

✅ **DO**:
- Start with the most obvious clusters
- Use Circle rules for round clusters
- Use Rectangle rules for box-shaped regions
- Use Axis thresholds for clear linear separations

❌ **DON'T**:
- Create overlapping rules (they conflict!)
- Make rules too small (might overfit)
- Make rules too large (misses precision)
- Ignore the visualization

### Improving Accuracy

**If accuracy is low (< 70%)**:
```
1. Check your rules - do they match the visual clusters?
2. Try different features (X and Y axes)
3. Add more rules for uncovered regions
4. Use Circle Segments to find better splits
```

**If accuracy is high but tree is complex**:
```
1. You might be overfitting
2. Try removing some rules
3. Combine multiple small rules into larger ones
```

---

## 🔧 Troubleshooting

### Common Issues

#### Problem: "I don't see any visualization"
**Solution**:
1. Make sure you selected a dataset
2. Check that X and Y features are selected
3. Try refreshing the page
4. Check browser console for errors

#### Problem: "My rule doesn't appear on the plot"
**Solution**:
1. Make sure you clicked "Add Rule" button
2. Check if rule parameters are within data range
3. Switch between tabs to refresh
4. Clear rules and try again

#### Problem: "Accuracy is 0%"
**Solution**:
1. Check that you assigned the correct class to rules
2. Verify rule parameters match the visual clusters
3. Make sure rules cover some data points
4. Try creating simpler rules first

#### Problem: "CSV upload fails"
**Solution**:
1. Ensure CSV has a column named 'class'
2. Check that file has numeric features
3. Verify CSV is properly formatted (no missing values)
4. Try one of the demo datasets first

#### Problem: "Application is slow"
**Solution**:
1. Use fewer features in Circle Segments
2. Reduce dataset size
3. Clear rules and start fresh
4. Close other browser tabs

---

## 🚀 Advanced Features

### Exporting Your Work

```python
# Coming soon: Export rules as JSON
# Your rules will be saved and can be reloaded later
```

### Using Custom Datasets

Your CSV must have:
- **Numeric columns**: Features for visualization
- **'class' column**: Target labels (e.g., "Setosa", "Healthy", "Type_A")

Example CSV structure:
```csv
feature1,feature2,feature3,class
5.1,3.5,1.4,Setosa
6.2,2.9,4.3,Versicolor
7.3,3.0,5.8,Virginica
```

### Splitting Strategy (from the paper)

The original PBC paper suggests this order:

1. **BPP (Best Pure Partitions)**: Find segments with one dominant color
2. **LCP (Largest Cluster Partitioning)**: Split large, pure clusters
3. **BCP (Best Complete Partitioning)**: Separate mixed regions
4. **DDP (Different Distribution Partitioning)**: Handle complex distributions

**Pro Tip**: Follow this order when creating rules!

---

## ❓ FAQs

### General Questions

**Q: Do I need programming experience?**
A: No! The interface is point-and-click. However, understanding basic statistics helps.

**Q: How is this different from regular machine learning?**
A: Regular ML is automatic. PBC lets YOU create the rules by looking at the data visually.

**Q: Can I use this for real projects?**
A: It's great for learning, exploring data, and prototyping. For production, combine insights with traditional ML.

**Q: What's the maximum dataset size?**
A: Works best with < 10,000 samples. Larger datasets may slow down visualization.

### Technical Questions

**Q: What's the difference between Scatter Plot and Circle Segments?**
A: 
- **Scatter Plot**: Shows 2 features in traditional X-Y plot
- **Circle Segments**: Shows ALL features in radial layout (from the research paper)

**Q: Can I create non-linear boundaries?**
A: Yes! Use Circle rules for curved boundaries. Combine multiple rules for complex shapes.

**Q: How do I know which features to select?**
A: Try different combinations! Look for features that separate classes well visually.

**Q: What if my rules overlap?**
A: The first matching rule wins. Order matters (currently chronological).

### Comparison Questions

**Q: Why is Decision Tree accuracy higher?**
A: It's optimized mathematically. But your PBC rules might be:
- Simpler (fewer rules)
- More interpretable
- Based on domain knowledge

**Q: Should I always aim for 100% accuracy?**
A: No! 70-90% with simple rules is often better than 95% with complex rules (avoids overfitting).

**Q: Can PBC beat Decision Trees?**
A: Sometimes! Especially when you have domain knowledge the algorithm doesn't.

---

## 📚 Additional Resources

### Research Paper
- **Title**: "Visual Classification: An Interactive Approach to Decision Tree Construction"
- **Authors**: Mihael Ankerst, Christian Elsen, Martin Ester, Hans-Peter Kriegel
- **Year**: 1999
- **Link**: [ResearchGate](https://www.researchgate.net/publication/221654449)

### Learning More

**Machine Learning Basics**:
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Decision Trees Explained](https://en.wikipedia.org/wiki/Decision_tree_learning)

**Data Visualization**:
- [Plotly Documentation](https://plotly.com/python/)
- [Streamlit Gallery](https://streamlit.io/gallery)

**Related Concepts**:
- Classification
- Decision Boundaries
- Feature Engineering
- Visual Data Mining

---

## 🎓 Practice Exercises

### Exercise 1: Perfect Classification
**Goal**: Achieve 100% accuracy on Iris dataset

**Steps**:
1. Load Iris dataset
2. Use X="petal length", Y="petal width"
3. Create exactly 3 rules (one per class)
4. Hint: These features separate the classes almost perfectly!

### Exercise 2: Minimal Rules Challenge
**Goal**: Get > 85% accuracy with only 2 rules

**Steps**:
1. Use Wine dataset
2. Experiment with different feature combinations
3. Find the two most discriminative boundaries
4. Tip: Use X-Axis or Y-Axis thresholds

### Exercise 3: Circle Segments Mastery
**Goal**: Use Circle Segments view to find optimal splits

**Steps**:
1. Select 6-8 features for visualization
2. Identify attributes with pure color segments
3. Create split rules based on visual inspection
4. Compare with scatter plot approach

### Exercise 4: Custom Dataset
**Goal**: Apply PBC to your own data

**Steps**:
1. Prepare a CSV with numeric features and 'class' column
2. Upload and explore
3. Create meaningful rules based on your domain knowledge
4. Compare with Decision Tree baseline

---

## 🆘 Getting Help

### If You're Stuck

1. **Re-read this guide** - especially the Step-by-Step Tutorial
2. **Start fresh** - Click "Clear All Rules" and try again
3. **Try a different dataset** - Some are easier than others
4. **Check the research paper** - For deeper understanding
5. **Experiment** - Learning by doing is the best way!

### Contact & Support

- **GitHub Issues**: [Report bugs or request features]
- **Email**: [Your contact email]
- **Community**: [Link to forum/Discord if available]

---

## 🎉 Conclusion

Congratulations! You now know how to:
- ✅ Load and visualize datasets
- ✅ Create classification rules visually
- ✅ Interpret accuracy and confusion matrices
- ✅ Compare manual vs automatic classification
- ✅ Use both Scatter Plot and Circle Segments views

**Remember**: The goal isn't always to beat the automatic algorithm. The goal is to:
- **Understand your data** deeply
- **Create interpretable rules** that make sense
- **Apply domain knowledge** that algorithms might miss

Now go forth and classify! 🌳✨

---

**Version**: 1.0  
**Last Updated**: 2025  
**Application**: PBC Visual Classification Dashboard  
**Based on**: Ankerst et al. (1999)

---

*Happy Classifying! 🎨📊🌳*