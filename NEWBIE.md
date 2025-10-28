# 🌳 PBC Visual Classification Dashboard - Complete User Guide

**Welcome to the Hybrid PBC Implementation!**  
*Combining authentic Circle Segments visualization with modern scatter plot interaction*

---

## 📚 Table of Contents

1. [Quick Start (5 Minutes)](#quick-start-5-minutes)
2. [Understanding PBC](#understanding-pbc)
3. [Interface Overview](#interface-overview)
4. [Complete Workflow Guide](#complete-workflow-guide)
5. [View Modes Explained](#view-modes-explained)
6. [Creating Rules](#creating-rules)
7. [Tree Operations](#tree-operations)
8. [Evaluation & Results](#evaluation--results)
9. [Advanced Techniques](#advanced-techniques)
10. [Troubleshooting](#troubleshooting)
11. [FAQs](#faqs)

---

## 🚀 Quick Start (5 Minutes)

### Installation & Launch

```bash
# Install dependencies
pip install streamlit pandas numpy plotly scikit-learn matplotlib

# Run the application
streamlit run pbc_dashboard.py
```

### Your First Classification (Iris Dataset)

**Step 1: Launch & Select Dataset**
```
1. App opens in browser automatically
2. Sidebar → Dataset: Select "Iris"
3. ✅ You now have 150 flower samples loaded
```

**Step 2: Choose View Mode**
```
1. Top-right dropdown → Select "Split Screen"
2. You'll see two windows side-by-side:
   - Left: Data Interaction Window (Circle Segments)
   - Right: Knowledge Interaction Window (Tree)
```

**Step 3: Create Your First Rule**
```
1. Sidebar → Rule Creation Method: "Circle Segments Splitline"
2. Select Attribute: "petal length (cm)"
3. Move slider to: ~2.5 (you'll see clear separation)
4. Assign Class: "setosa"
5. Direction: "Left (≤ split)"
6. Click: "✂️ Add Splitline Rule"
```

**Step 4: View Results**
```
1. Switch to "Tabbed" view
2. Go to "Evaluation" tab
3. See your accuracy: ~33% (one class covered!)
4. Add two more rules for other classes
5. Watch accuracy climb to 95%+! 🎉
```

**Congratulations!** You've completed your first PBC classification! 🌟

---

## 🎓 Understanding PBC

### What is PBC?

**PBC (Perception-Based Classification)** is a revolutionary approach where **YOU** build the decision tree by **visually exploring the data**, rather than letting an algorithm do it automatically.

### The Core Idea

```
Traditional ML:         PBC Approach:
┌──────────────┐       ┌──────────────┐
│ Feed Data    │       │ Visualize    │
│      ↓       │       │ Data First   │
│ Algorithm    │       │      ↓       │
│ Decides      │       │ You Explore  │
│      ↓       │       │ Patterns     │
│ Get Tree     │       │      ↓       │
│ (Black Box)  │       │ Create Rules │
│              │       │ Visually     │
│              │       │      ↓       │
│              │       │ Build Tree   │
│              │       │ (Transparent)│
└──────────────┘       └──────────────┘
```

### Key Advantages

1. **🧠 Domain Knowledge**: Use your expertise about the data
2. **👁️ Visual Understanding**: See patterns, don't just trust numbers
3. **🔄 Flexibility**: Non-binary splits, backtracking allowed
4. **📊 Interpretability**: Smaller, more understandable trees
5. **🎯 Interactive**: Real-time feedback on decisions

### The Circle Segments Innovation

From the original 1999 paper, this visualization technique maps multidimensional data to a circle:

```
        Attribute 3
             │
    Attr 2  │  Attr 4
         \  │  /
          \ │ /
    ───────●───────  Attribute 1
          / │ \
         /  │  \
    Attr 6  │  Attr 5
             │
        Attribute 7
```

- Each **segment** = one attribute
- Points **sorted** by attribute value
- **Colors** = class labels
- **Patterns** emerge visually!

---

## 🖥️ Interface Overview

### Main Layout Components

```
┌─────────────────────────────────────────────────────────────┐
│  🌳 PBC Dashboard        [Split Screen ▼]  [🌙 Theme]       │
├────────────┬────────────────────────────────────────────────┤
│            │                                                 │
│  SIDEBAR   │           MAIN CONTENT AREA                    │
│            │                                                 │
│  • Dataset │  Split Screen Mode:                            │
│  • Features│  ┌──────────────┬──────────────┐              │
│  • Node    │  │   Data       │  Knowledge   │              │
│  • Rules   │  │ Interaction  │ Interaction  │              │
│  • Ops     │  │   Window     │   Window     │              │
│            │  └──────────────┴──────────────┘              │
│            │                                                 │
│            │  OR Tabbed Mode:                                │
│            │  [Scatter] [Circle] [Eval] [Tree]              │
│            │                                                 │
└────────────┴─────────────────────────────────────────────────┘
```

### Sidebar Sections

#### 1. 📊 Dataset
- Choose from: Iris, Wine, Breast Cancer
- Or upload your own CSV

#### 2. 🎯 Features
- **Scatter Plot Features**: X and Y axes
- **Circle Segments Features**: Multiple attributes (4-12)

#### 3. 📍 Current Node
- Shows which tree node you're working on
- Displays samples, depth, status

#### 4. ✏️ Create Rules
- Two methods:
  - **Scatter Plot Rules**: Circles, rectangles, thresholds
  - **Circle Segments Splitline**: Interactive splitting

#### 5. 🌳 Tree Operations
- Undo, Clear All, Mark as Leaf

#### 6. 📋 Active Rules
- Summary of created rules

---

## 📋 Complete Workflow Guide

### The Full PBC Process (Step-by-Step)

---

### **PHASE 1: Dataset Selection & Exploration**

#### Step 1.1: Select Your Dataset

**Location**: Sidebar → Dataset section

**Options**:

| Dataset | Samples | Features | Classes | Difficulty |
|---------|---------|----------|---------|------------|
| Iris | 150 | 4 | 3 | ⭐ Easy |
| Wine | 178 | 13 | 3 | ⭐⭐ Medium |
| Breast Cancer | 569 | 30 | 2 | ⭐⭐⭐ Hard |
| Custom CSV | Varies | Varies | Varies | ⭐-⭐⭐⭐ |

**Recommendation**: Start with **Iris** - it's simple and has clear visual patterns.

**Action**:
```
1. Click dropdown: "Select Dataset"
2. Choose: "Iris"
3. ✅ Dataset loads automatically
4. You'll see: "150 samples loaded"
```

#### Step 1.2: Choose Features for Visualization

**Location**: Sidebar → Features section

**For Scatter Plot**:
```
1. X-Axis Feature: "sepal length (cm)"
2. Y-Axis Feature: "sepal width (cm)"
   
   OR try these powerful combinations:
   - X: "petal length (cm)", Y: "petal width (cm)" (best separation!)
   - X: "sepal length (cm)", Y: "petal length (cm)"
```

**For Circle Segments**:
```
1. Click: "Select Attributes"
2. Choose: 4-8 features (start with all 4 for Iris)
   - sepal length (cm)
   - sepal width (cm)
   - petal length (cm)
   - petal width (cm)
3. ✅ Selected features appear in circle visualization
```

**Why This Matters**: Different feature combinations reveal different patterns!

---

### **PHASE 2: Choose Your View Mode**

#### Step 2.1: Understanding View Modes

**Location**: Top-right dropdown

**Split Screen Mode** (Recommended for beginners):
- ✅ See data AND tree simultaneously
- ✅ Matches original PBC paper
- ✅ Best for understanding process
- Left: Data patterns
- Right: Growing tree structure

**Tabbed Mode** (For experienced users):
- ✅ More screen space per view
- ✅ Switch between different visualizations
- ✅ Better for detailed analysis

**Action**:
```
1. Click dropdown (top-right)
2. Select: "Split Screen"
3. Interface changes to dual-window layout
```

---

### **PHASE 3: Explore Your Data**

#### Step 3.1: Split Screen - Data Interaction Window

**Location**: Left side of screen

**What You See**:

**Option A: Circle Segments View** (Default)
```
Visual: Pizza-like circle divided into segments
- Each slice = one attribute
- Colors = class labels (Red, Blue, Green for Iris)
- Points arranged radially, sorted by value

What to look for:
✅ Segments with one dominant color (pure regions)
✅ Clear boundaries between colors
✅ Large homogeneous clusters
```

**Option B: Scatter Plot View**
```
Visual: Traditional 2D plot
- X-axis = chosen feature
- Y-axis = chosen feature
- Colored dots = samples (color = class)

What to look for:
✅ Separate clusters
✅ Clear gaps between groups
✅ Patterns that could be circled or boxed
```

**Switch Between Views**:
```
1. Data Interaction Window → Radio buttons
2. Select: "Circle Segments" or "Scatter Plot"
```

#### Step 3.2: Split Screen - Knowledge Interaction Window

**Location**: Right side of screen

**What You See**:

**Top Half: Tree Visualization**
```
Visual: Your decision tree growing
- Blue boxes = current node (where you're working)
- Yellow boxes = internal nodes (have children)
- Green boxes = leaf nodes (final classification)

Initially:
- Just one blue box: "Node 0 (150 samples)"
- This is your starting point!
```

**Bottom Half: Node Statistics**
```
Bar chart showing:
- How many samples of each class in current node
- Purity percentage
- Recommendations (e.g., "High purity! Consider leaf")

Example for Iris root node:
- Setosa: 50 samples (red bar)
- Versicolor: 50 samples (blue bar)  
- Virginica: 50 samples (green bar)
- Purity: 33.3% (all classes equal)
```

---

### **PHASE 4: Creating Your First Rule**

#### Step 4.1: Choose Rule Creation Method

**Location**: Sidebar → Create Rules section

**Two Methods Available**:

**Method A: Circle Segments Splitline** (Recommended First)
- Uses the Circle Segments visualization
- Pick attribute and split point
- True PBC paper method

**Method B: Scatter Plot Rules**
- Draw shapes on scatter plot
- More intuitive for beginners
- Multiple rule types

**Action**:
```
1. Rule Creation Method: Select "Circle Segments Splitline"
2. Continue to next step...
```

#### Step 4.2: Circle Segments Splitline Method

**Visual Guide**:

**What You're Doing**: Finding where to "cut" an attribute to separate classes

**Step-by-Step**:

```
Step 1: Select Attribute
└─ Sidebar → "Select Attribute"
└─ Choose: "petal length (cm)"
└─ Look at circle segments - find "petal length" segment

Step 2: Observe the Pattern
└─ In the petal length segment, you'll see:
   ├─ Red region (Setosa) at the start
   ├─ Blue region (Versicolor) in middle
   └─ Green region (Virginica) at the end

Step 3: Move the Slider
└─ "Split Point for petal length"
└─ Move to: ~2.5
└─ Watch the magnified view update

Step 4: Check Magnified View
└─ Expander: "🔍 Magnified Splitline View"
└─ Shows values near your split point
└─ Red dots (≤ 2.5) vs Blue dots (> 2.5)
└─ Vertical green line = your split

Step 5: Choose Direction & Class
└─ Direction: "Left (≤ split)"  ← Everything below 2.5
└─ Assign Class: "setosa"
└─ This means: "If petal length ≤ 2.5, classify as Setosa"

Step 6: Add the Rule!
└─ Click: "✂️ Add Splitline Rule"
└─ ✅ Success message appears
└─ Watch the tree update!
```

**What Happens Next**:
```
1. Right window (Knowledge) updates:
   - Node 0 now shows split
   - Two child nodes appear
   
2. Data Interaction shows updated data:
   - Now showing subset for current node
   - Used attribute (petal length) removed from circle

3. Sidebar shows:
   - "Current Node" updated
   - "Active Rules" count increases
```

#### Step 4.3: Scatter Plot Rules Method

**Alternative Approach** (If you prefer shapes):

**Rule Type: Circle** (for round clusters)
```
1. Look at scatter plot
2. See a round cluster? Note its center
3. Sidebar → Rule Type: "Circle"
4. Center X: (estimate from plot, e.g., 5.0)
5. Center Y: (estimate from plot, e.g., 3.5)
6. Radius: 1.5 (adjust as needed)
7. Assign Class: "setosa"
8. Click: "➕ Add Circle Rule"
```

**Rule Type: Rectangle** (for box-shaped regions)
```
1. Identify rectangular region
2. Rule Type: "Rectangle"
3. X Min: 4.5, X Max: 6.0
4. Y Min: 3.0, Y Max: 4.0
5. Assign Class: "setosa"
6. Click: "➕ Add Rectangle Rule"
```

**Rule Type: X-Axis Threshold** (vertical split)
```
1. Find vertical line that separates classes
2. Rule Type: "X-Axis Threshold"
3. Threshold: 5.5
4. Direction: "Left (≤)" or "Right (>)"
5. Assign Class: "setosa"
6. Click: "➕ Add X-Axis Rule"
```

**Rule Type: Y-Axis Threshold** (horizontal split)
```
1. Find horizontal line that separates classes
2. Rule Type: "Y-Axis Threshold"
3. Threshold: 3.0
4. Direction: "Below (≤)" or "Above (>)"
5. Assign Class: "virginica"
6. Click: "➕ Add Y-Axis Rule"
```

---

### **PHASE 5: Continue Building Your Tree**

#### Step 5.1: Add Rules for Other Classes

**You've classified one class, now classify the rest!**

**Iris Example** (after first rule):

```
Rule 1: petal length ≤ 2.5 → Setosa ✅
Remaining: Versicolor and Virginica (need more rules!)

Look at the data:
- Current node now shows only 100 samples (50 Versicolor, 50 Virginica)
- Setosa is gone (already classified!)

Rule 2: Create another split
1. Select Attribute: "petal width (cm)"
2. Move slider to: ~1.7
3. Direction: "Left (≤ split)"
4. Assign Class: "versicolor"
5. Add rule

Rule 3: What's left?
- Remaining samples are mostly Virginica
- You can mark this as a leaf!
```

#### Step 5.2: Navigate Tree Nodes

**After creating splits, you have multiple nodes**:

```
Tree Structure:
         Node 0 (Root)
         /           \
    Node 1         Node 2
  (Setosa)    (Ver + Vir)
                /        \
           Node 3      Node 4
        (Versicolor) (Virginica)
```

**How to Navigate**:
```
Currently: You're always working on the "current node" (blue box)

To work on a different node:
└─ Currently: Manual navigation not implemented
└─ Automatic: System moves you to unfinished nodes

To mark a node complete:
└─ Sidebar → "Mark as Leaf"
└─ Assigns majority class to that node
└─ Node turns green in tree
```

#### Step 5.3: Understand "Used Attributes"

**Important Concept**:

```
Rule: You can't use the same attribute twice in one path!

Example:
If you split on "petal length" at root:
└─ Child nodes CANNOT split on "petal length" again
└─ Circle segments will show fewer attributes
└─ This prevents redundant splits

Why?
└─ Once you split on an attribute, that decision is made
└─ Children inherit that constraint
└─ Forces you to use different features
```

**Visual Indicator**:
```
Sidebar → Current Node → "Used Attributes:"
  • petal length (cm)

Circle Segments:
└─ "petal length" segment will disappear
└─ Only remaining attributes shown
```

---

### **PHASE 6: Evaluation & Results**

#### Step 6.1: Check Your Accuracy

**Switch to Tabbed View** (if not already):
```
1. Top-right: Select "Tabbed"
2. Click tab: "📈 Evaluation"
```

**What You See**: Two side-by-side comparisons

**Left Side: Your PBC Model**
```
┌─────────────────────┐
│ 🎨 PBC Model        │
│                     │
│     95.3%          │ ← Your accuracy!
│                     │
│ Rules: 3            │
│ Tree Nodes: 7       │
└─────────────────────┘

Confusion Matrix:
Shows how many correct/incorrect per class

Classification Report:
Precision, Recall, F1-Score for each class
```

**Right Side: sklearn Decision Tree**
```
┌─────────────────────┐
│ 🌳 Decision Tree    │
│                     │
│     96.0%          │ ← Baseline
│                     │
│ Depth: 5            │
│ Nodes: 15           │
└─────────────────────┘
```

#### Step 6.2: Interpret Results

**Accuracy Scores**:

| Range | Interpretation | Action |
|-------|----------------|--------|
| < 50% | Poor - rules don't match patterns | Redo rules |
| 50-70% | Fair - basic patterns captured | Add more rules |
| 70-85% | Good - most patterns captured | Fine-tune |
| 85-95% | Excellent - competitive! | Maybe done! |
| > 95% | Outstanding - excellent work! | Check for overfitting |

**Confusion Matrix Reading**:
```
Example:
                Predicted
              Set  Ver  Vir
Actual  Set [ 50   0    0 ] ← Perfect!
        Ver [  0  47    3 ] ← 3 mistakes
        Vir [  0   2   48 ] ← 2 mistakes

Diagonal = Correct ✅
Off-diagonal = Errors ❌

Total Errors: 3 + 2 = 5 out of 150 = 96.7% accuracy
```

**Classification Report**:
```
              precision  recall  f1-score
setosa           1.00     1.00     1.00  ← Perfect!
versicolor       0.96     0.94     0.95  ← Great
virginica        0.94     0.96     0.95  ← Great

Precision: "When I say X, am I usually right?"
Recall: "Of all X samples, how many did I find?"
F1-Score: Average (higher = better)
```

#### Step 6.3: Compare with Decision Tree

**Model Comparison**:

```
✅ PBC Advantages:
- Fewer rules (3 vs 15 nodes)
- More interpretable
- You understand WHY
- Used domain knowledge

✅ Decision Tree Advantages:
- Slightly higher accuracy
- Fully automatic
- Optimized mathematically
```

**Success Criteria**:

You're doing GREAT if:
- ✅ Within 5% of Decision Tree accuracy
- ✅ Fewer rules than Decision Tree nodes
- ✅ You understand each rule you created

---

### **PHASE 7: View Your Tree Structure**

#### Step 7.1: PBC Tree Visualization

**Navigate to**:
```
Tabbed View → "🌳 Decision Tree" tab
```

**What You See**:

```
Left Side: Your PBC Tree
┌─────────────────────────────────┐
│         Node 0 (150)            │ ← Blue = current
│      petal length ≤ 2.5         │
│         /          \            │
│   Node 1         Node 2         │ ← Green = leaf
│  Setosa (50)   (100 samples)    │
│                 /      \        │
│           Node 3      Node 4    │
│        Versicolor  Virginica    │
└─────────────────────────────────┘

Right Side: Tree Statistics
- Total Nodes: 7
- Leaf Nodes: 3
- Internal Nodes: 4
- Max Depth: 2
- Active Rules: 3

Node Details (expandable):
Each node shows:
├─ Samples count
├─ Depth level
├─ Split attribute (if any)
├─ Class label (if leaf)
└─ Used attributes
```

#### Step 7.2: Compare Trees

**Below**: sklearn Decision Tree shown for comparison

```
Visual:
- Much larger tree (15 nodes)
- More complex structure
- Harder to interpret

But:
- May be slightly more accurate
- Automatically optimized
```

**Takeaway**:
```
Your PBC tree is:
✅ Smaller
✅ Clearer
✅ Meaningful to you
✅ Based on visual patterns

sklearn tree is:
✅ Optimized
✅ Automatic
❌ Black box
❌ Less interpretable
```

---

## 🎯 View Modes Explained

### Split Screen Mode (Recommended)

**Best For**:
- Learning PBC methodology
- Following the research paper
- Understanding the process
- Real-time feedback

**Layout**:
```
┌─────────────────┬─────────────────┐
│  DATA           │  KNOWLEDGE      │
│  INTERACTION    │  INTERACTION    │
│                 │                 │
│  [Viz Mode]     │  [Tree]         │
│                 │                 │
│  Circle         │    Node 0       │
│  Segments       │    /    \       │
│  or Scatter     │  N1      N2     │
│                 │                 │
│                 │  [Stats]        │
│                 │  Samples: 150   │
│                 │  Purity: 33%    │
└─────────────────┴─────────────────┘
```

**Workflow**:
```
1. Look at data (left)
2. Identify pattern
3. Create rule (sidebar)
4. Watch tree grow (right)
5. See statistics update
6. Repeat!
```

### Tabbed Mode (Advanced)

**Best For**:
- Detailed analysis
- More screen space
- Experienced users
- Focused work

**Tabs**:

**Tab 1: Scatter Plot View**
- Full-screen 2D visualization
- All rules overlaid
- Interactive zooming
- Rule summary below

**Tab 2: Circle Segments View**
- Full-screen radial layout
- PBC splitting strategy guidance
- Attribute analysis tools
- Suggested split points

**Tab 3: Evaluation**
- Performance metrics
- Confusion matrices
- Comparison with sklearn
- Detailed insights

**Tab 4: Decision Tree**
- Your tree structure
- Statistics panel
- Node details
- sklearn comparison

---

## ✏️ Creating Rules

### Rule Creation Methods Summary

| Method | View | Best For | Difficulty |
|--------|------|----------|------------|
| Circle Splitline | Circle Segments | Multi-dimensional | ⭐⭐ |
| Circle | Scatter Plot | Round clusters | ⭐ |
| Rectangle | Scatter Plot | Box regions | ⭐ |
| X-Threshold | Scatter Plot | Vertical splits | ⭐ |
| Y-Threshold | Scatter Plot | Horizontal splits | ⭐ |

### Detailed Rule Creation

#### Circle Segments Splitline

**When to Use**:
- Working with many features (>2)
- Need to consider all dimensions
- Following PBC paper methodology

**How It Works**:
```
1. Each attribute shown in circle segment
2. Data sorted by attribute value
3. Colors show class distribution
4. You pick where to "cut"

Visual:
   Red Red Red | Blue Blue Green Green
   ←─────────→
      Split here at value 2.5
```

**Steps**:
```
1. Select attribute from dropdown
2. Observe color distribution in segment
3. Move slider to find good separation
4. Check magnified view for exact values
5. Choose direction (left/right of split)
6. Assign class to that side
7. Add rule
```

**Tips**:
```
✅ DO:
- Look for clear color transitions
- Use magnified view to verify
- Check multiple attributes before deciding

❌ DON'T:
- Split in middle of same-color region
- Ignore the visualization
- Rush the decision
```

#### Scatter Plot Rules

**Circle Rules**:
```
Purpose: Capture round, cluster-like groups

Example: Iris Setosa
- Forms tight circular cluster
- Center around (5.0, 3.5)
- Radius 1.0 covers most points

Parameters:
├─ Center X: Where center is horizontally
├─ Center Y: Where center is vertically
└─ Radius: How big the circle is

Visualization:
      ●●●
     ●●●●●  ← Dashed circle appears on plot
      ●●●
```

**Rectangle Rules**:
```
Purpose: Capture box-shaped regions

Example: Middle cluster
- Forms rectangular region
- Boundaries: X[5.5-7.0], Y[2.0-3.5]

Parameters:
├─ X Min: Left edge
├─ X Max: Right edge
├─ Y Min: Bottom edge
└─ Y Max: Top edge

Visualization:
┌────────┐
│ ●●●●● │ ← Dashed rectangle on plot
│ ●●●●● │
└────────┘
```

**Axis Threshold Rules**:
```
Purpose: Simple linear splits

X-Axis (Vertical line):
    │ ●●●
●●●●│
    │

Y-Axis (Horizontal line):
●●●●●
─────
 ●●●

Parameters:
├─ Threshold: Where to draw line
└─ Direction: Which side gets the class

Use when:
- Clear linear separation visible
- One feature dominates
- Simple split works well
```

### The PBC Splitting Strategy

**From the Research Paper** - Use this order:

**1. BPP (Best Pure Partitions)**
```
Definition: Segment has one dominant color

Look for:
- Entire segment is mostly one color
- Clear homogeneous region
- High purity

Action:
- Split to isolate this pure region
- Mark as leaf immediately

Example:
Segment: [Red Red Red Red Red] (all setosa)
Action: Split and mark as Setosa leaf
```

**2. LCP (Largest Cluster Partitioning)**
```
Definition: Large cluster of same color

Look for:
- Big group of one color
- Not necessarily pure, but dominant
- Worth isolating

Action:
- Split to separate this cluster
- May need more splits on remainder

Example:
Segment: [Red Red Red Blue Blue] (mostly red)
Action: Split at red-blue boundary
```

**3. BCP (Best Complete Partitioning)**
```
Definition: Can separate into multiple regions

Look for:
- Multiple distinguishable groups
- Each with dominant color
- Multiple split points beneficial

Action:
- Create multiple splits
- Advantage: Non-binary splits allowed!

Example:
Segment: [Red Red | Blue Blue | Green Green]
Action: Two splits, creates three regions
```

**4. DDP (Different Distribution Partitioning)**
```
Definition: Distributions clearly differ

Look for:
- Mixed colors but different patterns
- No pure regions
- Statistical differences

Action:
- Find best discriminating point
- Even mixed, one side may be "more X"

Example:
Segment: [R B R B | G B G] (left more R/B, right more G/B)
Action: Split even if both sides mixed
```

**Priority Order**: Try BPP first, then LCP, then BCP, then DDP

---

## 🌳 Tree Operations

### Undo Last Action

**Purpose**: Revert your last rule

**Location**: Sidebar → Tree Operations → "↩️ Undo Last"

**What Happens**:
```
Before:
Rules: [Rule1, Rule2, Rule3]

Click "Undo Last"

After:
Rules: [Rule1, Rule2]

Tree updates automatically
Can undo multiple times
```

**Use When**:
- Made a mistake
- Want to try different rule
- Accuracy decreased after last rule

### Clear All Rules

**Purpose**: Start fresh

**Location**: Sidebar → Tree Operations → "🗑️ Clear All"

**What Happens**:
```
Resets everything:
├─ All rules deleted
├─ Tree reset to root node
├─ Current node = Node 0
└─ Ready to start over
```

**Use When**:
- Want to try completely different approach
- Made too many mistakes
- Learning/practicing

**Warning**: Cannot be undone!

### Mark as Leaf

**Purpose**: Stop splitting this node

**Location**: Sidebar → Tree Operations → "🍃 Mark as Leaf"

**When Available**: Only when current node is not already a leaf

**What Happens**:
```
1. Current node marked as terminal
2. Assigned the class you selected
3. Node turns green in tree
4. No more splitting on this path
5. System may move you to next unfinished node
```

**Use When**:
```
✅ Node has high purity (>90%)
✅ Only one class dominates heavily  
✅ Further splitting won't help
✅ You're satisfied with this branch

Example:
Node 3: 48 Setosa, 2 Versicolor
└─ 96% purity → Mark as Setosa leaf
```

**Strategy**:
```
Don't obsess over 100% purity!

Good: 90%+ purity → mark as leaf
Risk: Over-splitting → worse generalization
```

---

## 📊 Evaluation & Results

### Understanding Accuracy

**What is Accuracy?**
```
Accuracy = (Correct Predictions / Total Samples) × 100%

Example:
150 samples total
143 predicted correctly
7 predicted wrong

Accuracy = 143/150 = 95.3%
```

**Interpreting Your Score**:

```
Your 85% vs sklearn's 87%:
✅ Great! Within 2% is excellent for manual approach
✅ Your tree probably has fewer rules
✅ More interpretable
✅ You understand every decision

Your 95% vs sklearn's 87%:
🎉 Outstanding! You beat the algorithm!
🎉 Your domain knowledge paid off
🎉 Visual approach found better patterns
```

### Reading the Confusion Matrix

**What It Shows**: Where your model makes mistakes

**Example Matrix**:
```
                    Predicted
                Set    Ver    Vir
Actual  Set  [  50     0      0  ]  ← Row 1
        Ver  [   0    47      3  ]  ← Row 2
        Vir  [   0     2     48  ]  ← Row 3
         ↑     ↑      ↑      ↑
       Col 1  Col 2  Col 3
```

**How to Read**:

**Diagonal (Perfect Predictions)**:
```
[50, 47, 48] ← Correct classifications

Row 1, Col 1: 50 Setosa correctly predicted as Setosa ✅
Row 2, Col 2: 47 Versicolor correctly as Versicolor ✅
Row 3, Col 3: 48 Virginica correctly as Virginica ✅
```

**Off-Diagonal (Errors)**:
```
Row 2, Col 3: 3 ← 3 Versicolor wrongly predicted as Virginica ❌
Row 3, Col 2: 2 ← 2 Virginica wrongly predicted as Versicolor ❌

Why?
- These classes are similar
- Overlapping features
- Hard to separate perfectly
```

**Analysis Questions**:
```
1. Which class has most errors?
   → Look at the row with most off-diagonal values

2. Which classes confuse each other?
   → Look for symmetry (Ver→Vir and Vir→Ver)

3. Is any class perfect?
   → Look for rows with only diagonal value
   → Setosa usually perfect in Iris!
```

### Classification Report Explained

**Metrics for Each Class**:

```
              precision  recall  f1-score  support
setosa           1.00     1.00     1.00       50
versicolor       0.96     0.94     0.95       50
virginica        0.94     0.96     0.95       50
```

**Precision**: "When I predict class X, how often am I right?"
```
Formula: True Positives / (True Positives + False Positives)

Example - Versicolor:
- Predicted Versicolor 49 times
- 47 were actually Versicolor
- 2 were actually Virginica
- Precision = 47/49 = 0.96 = 96%

Interpretation:
- High precision = Few false alarms
- "When I say Versicolor, I'm usually right"
```

**Recall**: "Of all actual class X, how many did I find?"
```
Formula: True Positives / (True Positives + False Negatives)

Example - Versicolor:
- Actually 50 Versicolor samples
- Found 47 of them
- Missed 3 (called them Virginica)
- Recall = 47/50 = 0.94 = 94%

Interpretation:
- High recall = Found most of them
- "I found most Versicolor samples"
```

**F1-Score**: "Overall performance for this class"
```
Formula: 2 × (Precision × Recall) / (Precision + Recall)

Example - Versicolor:
- F1 = 2 × (0.96 × 0.94) / (0.96 + 0.94)
- F1 = 0.95 = 95%

Interpretation:
- Balanced measure
- Closer to 1.00 = better
- Average of precision and recall (harmonic mean)
```

**Support**: "How many samples of this class?"
```
Support = Total actual samples in dataset

All classes have 50 for Iris (balanced dataset)
```

### Model Comparison Insights

**Interpretability Winner**:
```
Measure: Number of nodes/rules

Your PBC: 3 rules, 7 nodes
sklearn: 15 nodes

Winner: PBC! 🎉
- Fewer rules = easier to explain
- Each rule makes sense to you
- Can justify every decision
```

**Accuracy Winner**:
```
Measure: Percentage correct

Your PBC: 95.3%
sklearn: 96.0%

Winner: sklearn (slightly)
- Only 0.7% difference
- Negligible in practice
- PBC is more interpretable though!
```

**Overall Assessment**:
```
Best choice depends on goal:

For deployment: sklearn (slightly more accurate)
For understanding: PBC (much clearer)
For teaching: PBC (shows the process)
For trust: PBC (you know why)
```

---

## 🎓 Advanced Techniques

### Multi-Way Splits (Non-Binary)

**Power of PBC**: Unlike traditional trees, you can split into MORE than 2 branches!

**Example**:
```
Traditional (Binary):
        petal_length ≤ 2.5?
          /          \
        Yes           No
    (Setosa)    (still mixed)

PBC (Multi-Way):
        petal_length
       /      |      \
    ≤2.5   2.5-4.5   >4.5
   Setosa  Versi   Virgi
```

**How to Create**:
```
1. Select attribute
2. Create MULTIPLE splitline rules:
   - Rule 1: petal_length ≤ 2.5 → Setosa
   - Rule 2: petal_length ≤ 4.5 → Versicolor
   - Rule 3: petal_length > 4.5 → Virginica

Result: 3-way split instead of binary!
```

**Advantages**:
- Fewer tree levels
- More efficient
- Clearer interpretation

### Backtracking Strategy

**When Things Don't Work Out**:

```
Scenario:
1. Split on attribute A
2. Accuracy: 70%
3. Split on attribute B (child node)
4. Accuracy: 65% ← Got worse!

Problem: Attribute A wasn't the best first choice

Solution: Backtrack!
```

**How to Backtrack**:
```
1. Click "↩️ Undo Last" (removes bad rule)
2. Click again (goes back to attribute A split)
3. Try different attribute or split point
4. Check if accuracy improves

Keep undoing until you find better path!
```

**Best Practice**:
```
Before Committing:
1. Try split mentally
2. Estimate likely accuracy
3. If unsure, try it
4. Check accuracy
5. Undo if worse
6. Try alternative

Learn from mistakes!
```

### Handling Imbalanced Data

**Problem**: Some classes have many samples, others few

**Example**:
```
Class A: 400 samples
Class B: 80 samples
Class C: 20 samples

Risk: Model ignores small classes
```

**Strategies**:

**1. Visual Inspection**:
```
Circle Segments view:
- Small classes show as thin color strips
- Easy to miss in visualization
- Zoom in mentally on these regions
```

**2. Priority to Small Classes**:
```
Create rules for rare classes FIRST:
1. Find Class C (rare) regions
2. Create precise rules
3. Then handle Class B
4. Finally Class A (easy, it's everywhere)

Why? Small classes need special attention
```

**3. Purity Threshold Adjustment**:
```
For rare classes:
- Accept lower purity (70% instead of 90%)
- Better to capture some than miss all
- Monitor precision vs recall trade-off
```

### Feature Engineering Tips

**Choosing Best Features**:

**Method 1: Try Different Combinations**
```
For Iris:
- sepal_length × sepal_width
- petal_length × petal_width ← Usually best!
- sepal_length × petal_length
- ... try all pairs

Best: Features that separate classes visually
```

**Method 2: Circle Segments Analysis**
```
Look at each segment:
1. Which shows clearest color separation?
2. Which has largest pure regions?
3. Which follows BPP strategy?

Use those attributes first!
```

**Method 3: Correlation Analysis**
```
Sidebar → Attribute Analysis:
1. Select attribute
2. View sorted display
3. See class ranges
4. Non-overlapping ranges = good feature!
```

### Working with High-Dimensional Data

**Challenge**: 30+ features (like Breast Cancer dataset)

**Strategies**:

**1. Feature Selection**:
```
Don't use all 30 at once!

Select 6-8 most important:
- Use circle segments with subset
- Focus on discriminative features
- Add more only if needed
```

**2. Multiple Views**:
```
View 1: Features 1-6 (circle segments)
View 2: Features 7-12 (circle segments)
View 3: Best pair (scatter plot)

Switch between views to understand all dimensions
```

**3. Hierarchical Approach**:
```
Level 1: Use 2-3 features to separate obvious groups
Level 2: For remaining, use different 2-3 features
Level 3: Fine-tune with more features

Gradually increase complexity
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Issue 1: "I don't see any patterns in Circle Segments"

**Possible Causes**:
```
1. Too many attributes selected (circle too crowded)
2. Attributes not discriminative
3. Need to change view angle mentally
```

**Solutions**:
```
✅ Reduce to 4-6 attributes
✅ Try different attribute combinations
✅ Switch to Scatter Plot view temporarily
✅ Use "Attribute Analysis" tool
✅ Look for segments with color gradients (not mixed)
```

#### Issue 2: "My accuracy is stuck at 50-60%"

**Diagnosis**:
```
Check:
1. How many rules created? (may need more)
2. Rules overlapping? (may conflict)
3. Features chosen? (may be poor)
4. Rule parameters? (may be off)
```

**Solutions**:
```
✅ Create more specific rules
✅ Try different feature pairs
✅ Check scatter plot for visual clusters
✅ Clear all and start fresh
✅ Follow splitting strategy (BPP → LCP → BCP → DDP)
```

#### Issue 3: "Rules not appearing on visualization"

**Possible Causes**:
```
1. Rule parameters outside data range
2. Rule type doesn't match view
3. Browser rendering issue
```

**Solutions**:
```
✅ Check rule parameters in sidebar
✅ Verify data range (min/max)
✅ Refresh page (F5)
✅ Switch between tabs
✅ Try different rule type
```

#### Issue 4: "Cannot add more rules - all attributes used"

**Understanding**:
```
You've used all attributes in this tree path!

Example:
Root: Split on petal_length
Child: Split on petal_width
Child: Split on sepal_length
Child: Split on sepal_width
Now: No attributes left!

This is normal and correct behavior.
```

**Solutions**:
```
✅ Mark current node as leaf
✅ Backtrack to try different split
✅ Accept this as terminal node
✅ Assign majority class

You cannot reuse attributes on same path!
```

#### Issue 5: "Tree visualization not updating"

**Causes**:
```
1. Need to refresh
2. State synchronization issue
3. Browser cache
```

**Solutions**:
```
✅ Switch views (Split Screen ↔ Tabbed)
✅ Refresh browser page
✅ Clear browser cache
✅ Restart Streamlit app
✅ Check console for errors (F12)
```

#### Issue 6: "Magnified view shows confusing colors"

**Understanding**:
```
Magnified view shows:
- Red dots: Values ≤ split (left side)
- Blue dots: Values > split (right side)
- NOT class colors!

This is intentional to show the split effect.
```

**What to Look For**:
```
Good split: Clear separation of class labels in hover text
Bad split: Mixed classes on both sides

Example:
Split at 2.5
Left (red dots): Hover shows "Setosa, Setosa, Setosa" ✅
Right (blue dots): Hover shows "Vers, Virg, Virg" ✅

vs

Split at 3.0
Left: "Setosa, Vers, Setosa" ❌ Mixed!
Right: "Vers, Virg, Setosa" ❌ Mixed!
```

### Performance Issues

#### Slow Visualization

**Symptoms**:
```
- Circle segments takes long to load
- Scatter plot laggy
- UI freezing
```

**Solutions**:
```
✅ Reduce circle segment features (6 max)
✅ Use smaller dataset
✅ Close other browser tabs
✅ Reduce browser zoom level
✅ Use Tabbed mode (less simultaneous rendering)
```

#### App Crashes

**Common Causes**:
```
1. Too large dataset (>50K samples)
2. Memory overflow
3. Invalid CSV format
```

**Solutions**:
```
✅ Sample your data (use first 10K rows)
✅ Check CSV format (needs 'class' column)
✅ Ensure numeric features only
✅ Restart Streamlit
✅ Check system RAM
```

### Data Issues

#### CSV Upload Fails

**Requirements**:
```
Your CSV must have:
✓ Numeric columns (features)
✓ One 'class' column (target)
✓ No missing values
✓ Reasonable size (<50MB)
```

**Example Valid CSV**:
```csv
feature1,feature2,feature3,class
5.1,3.5,1.4,Setosa
6.2,2.9,4.3,Versicolor
7.3,3.0,5.8,Virginica
```

**Common Errors**:
```
❌ No 'class' column → Add one
❌ Text in numeric columns → Convert to numbers
❌ Missing values → Fill or remove rows
❌ Too many classes (>10) → May need simplification
```

---

## ❓ FAQs

### General Questions

**Q: How is this different from normal machine learning?**

A: Traditional ML is fully automatic - algorithm decides everything. PBC is interactive - YOU make decisions based on visual patterns. You're in control!

**Q: Do I need to know programming?**

A: No! It's point-and-click. But basic statistics knowledge helps understand metrics.

**Q: Can I use this for real projects?**

A: Yes for:
- Exploratory data analysis
- Understanding your data deeply
- Prototyping classification approaches
- Teaching/learning ML concepts
- Small-medium datasets

Maybe not for:
- Production systems with huge data
- When you need automatic updates
- Real-time classification
- When accuracy is critical

**Q: What's the best dataset to start with?**

A: **Iris** - it's small, clean, and has clear visual patterns. Perfect for learning!

**Q: How long does it take to build a tree?**

A: 
- Simple dataset (Iris): 5-10 minutes
- Medium dataset (Wine): 15-30 minutes
- Complex dataset (Breast Cancer): 30-60 minutes

Depends on your experience level too!

### Technical Questions

**Q: Why can't I use the same attribute twice?**

A: This prevents redundant splits. Once you split on "petal_length", that decision is made for all child nodes. Use different attributes for further refinement.

**Q: What's the maximum number of features I can visualize?**

A: 
- Circle Segments: 4-12 features optimal
- Scatter Plot: 2 features (but you can switch pairs)
- Too many features makes visualization crowded

**Q: Can I create non-binary splits?**

A: YES! This is a major advantage of PBC. Create multiple rules on same attribute with different thresholds.

**Q: How do I know which splitting strategy to use (BPP, LCP, etc.)?**

A: 
1. Try BPP first (pure regions)
2. If none, try LCP (large clusters)
3. If none, try BCP (complete partition)
4. Last resort: DDP (different distributions)

Follow this order!

**Q: Can I export my tree?**

A: Currently not directly, but you can:
- Screenshot the tree visualization
- Copy rule parameters from sidebar
- Save confusion matrix results
- Export functionality coming soon!

**Q: What's the difference between "Split Screen" and "Tabbed" views?**

A:
- **Split Screen**: Data + Tree visible together (like paper)
- **Tabbed**: More space per view, switch between them

Try both, use what feels better!

### Comparison Questions

**Q: Will my PBC model always be worse than sklearn?**

A: No! Often comparable (within 5%). Sometimes better if you have domain knowledge the algorithm doesn't!

**Q: My tree has fewer nodes but same accuracy. Is this bad?**

A: No, it's GREAT! Fewer nodes = more interpretable = better generalization. This is a PBC advantage!

**Q: Should I aim for 100% accuracy?**

A: No! 85-95% is usually great. 100% might mean overfitting (memorizing data, not learning patterns).

**Q: Why does sklearn tree have more nodes?**

A: It aggressively optimizes accuracy, creating complex trees. PBC balances accuracy with interpretability.

### Strategy Questions

**Q: How many rules should I create?**

A: No fixed number. Typical:
- 3-5 rules for simple datasets (Iris)
- 5-10 rules for medium datasets (Wine)
- 10-20 rules for complex datasets (Breast Cancer)

Stop when accuracy is satisfactory!

**Q: When should I mark a node as leaf?**

A: When:
- Purity > 90%
- Further splitting won't help much
- All attributes used
- You're satisfied with this branch

**Q: What if classes overlap heavily?**

A: 
- Accept that 100% separation impossible
- Aim for "mostly correct"
- Focus on high-purity regions first
- Use multiple features to disambiguate

**Q: Should I always use circle segments or scatter plot?**

A: Use both!
- **Circle Segments**: When working with many features
- **Scatter Plot**: When focusing on 2 features, drawing shapes
- **Switch between them** for full understanding

---

## 🎓 Practice Exercises

### Exercise 1: Perfect Classification (Beginner)

**Goal**: Achieve >95% accuracy on Iris with ≤3 rules

**Steps**:
```
1. Load Iris dataset
2. Use petal_length and petal_width features
3. Split Screen view
4. Create 3 splitline rules:
   - petal_length ≤ 2.5 → Setosa
   - petal_width ≤ 1.7 → Versicolor
   - Remainder → Virginica
5. Check accuracy in Evaluation tab

Target: >95% accuracy
```

**Learning Objectives**:
- Basic interface navigation
- Splitline rule creation
- Reading evaluation metrics

### Exercise 2: Minimal Rules Challenge (Intermediate)

**Goal**: Get >85% accuracy with ONLY 2 rules

**Steps**:
```
1. Load Wine dataset
2. Experiment with different feature pairs
3. Find the two most discriminative splits
4. Hint: Use axis threshold rules
5. Compare with sklearn (it uses many more nodes!)

Target: >85% with 2 rules
```

**Learning Objectives**:
- Feature selection importance
- Rule efficiency
- Trade-off between rules and accuracy

### Exercise 3: Multi-Way Splits (Advanced)

**Goal**: Create a 3-way split on one attribute

**Steps**:
```
1. Load Iris dataset
2. Create 3 rules on petal_length:
   - ≤ 2.5 → Setosa
   - 2.5-4.5 → Versicolor  
   - > 4.5 → Virginica
3. Compare tree depth with binary approach
4. Observe tree structure difference

Learning: Non-binary splits create flatter trees
```

**Learning Objectives**:
- Multi-way splitting
- Tree structure optimization
- PBC advantages over traditional methods

### Exercise 4: Circle Segments Mastery (Advanced)

**Goal**: Use ONLY circle segments view (no scatter plot)

**Steps**:
```
1. Load Wine dataset (13 features)
2. Select 8 features for circle segments
3. Identify pure segments using BPP strategy
4. Create all rules using splitline method
5. Never switch to scatter plot!

Target: >80% accuracy
```

**Learning Objectives**:
- High-dimensional visualization
- Splitting strategy application
- Circle segments interpretation

### Exercise 5: Real-World Application (Expert)

**Goal**: Apply to your own dataset

**Steps**:
```
1. Prepare your CSV:
   - Numeric features
   - 'class' column
   - No missing values
2. Upload to application
3. Explore with both views
4. Apply splitting strategies
5. Document your process
6. Compare with sklearn baseline

Reflect: Did visual approach reveal insights?
```

**Learning Objectives**:
- Real-world application
- Data preparation
- Process documentation
- Critical analysis

---

## 📚 Additional Resources

### Research Paper

**Original Paper**:
- Title: "Visual Classification: An Interactive Approach to Decision Tree Construction"
- Authors: Mihael Ankerst, Christian Elsen, Martin Ester, Hans-Peter Kriegel
- Year: 1999
- Conference: KDD '99
- [that paper link](./paper.pdf)

**Key Contributions**:
1. Circle Segments visualization technique
2. Interactive decision tree construction
3. Non-binary splits for numeric attributes
4. Backtracking capability
5. Human-in-the-loop machine learning

### Related Concepts

**Machine Learning**:
- Decision Trees
- Classification algorithms
- Feature engineering
- Model evaluation metrics

**Visualization**:
- Multidimensional data visualization
- Pixel-oriented techniques
- Interactive graphics
- Visual analytics

**Human-Computer Interaction**:
- Visual decision making
- Interactive machine learning
- Explainable AI
- User-centered design

### Learning Path

**Beginner**:
1. Complete Quick Start (5 min)
2. Do Exercise 1 (Perfect Classification)
3. Read "Understanding PBC" section
4. Practice with Iris dataset
5. Learn evaluation metrics

**Intermediate**:
1. Try all three demo datasets
2. Complete Exercises 2-3
3. Master both view modes
4. Learn splitting strategies
5. Compare with sklearn consistently

**Advanced**:
1. Complete Exercises 4-5
2. Apply to custom datasets
3. Optimize for fewer rules
4. Document best practices
5. Teach others!

---

## 🎉 Conclusion

Congratulations! You now have complete knowledge of the PBC Visual Classification Dashboard!

### What You've Learned

✅ **Core Concepts**:
- PBC methodology
- Circle Segments visualization
- Interactive tree construction
- Splitting strategies

✅ **Practical Skills**:
- Dataset selection and loading
- Feature configuration
- Rule creation (both methods)
- Tree navigation and operations
- Result interpretation

✅ **Advanced Techniques**:
- Multi-way splits
- Backtracking
- Imbalanced data handling
- High-dimensional visualization

### Next Steps

1. **Practice**: Work through all 5 exercises
2. **Experiment**: Try different datasets
3. **Share**: Teach someone else
4. **Apply**: Use on real problems
5. **Contribute**: Suggest improvements!

### Remember

The goal of PBC isn't always to beat automatic algorithms in accuracy. The goal is to:

🎯 **Understand your data** deeply  
🎯 **Create interpretable** models  
🎯 **Apply domain knowledge** effectively  
🎯 **Learn through interaction**  
🎯 **Build trust** in your models  

---

**Happy Classifying!** 🌳✨

*Version: 2.0 Hybrid Implementation*  
*Last Updated: 2025*  
*Based on: Ankerst et al. (1999)*

---

## 📞 Support & Feedback

Having issues? Have suggestions?

- **Report Bugs**: [GitHub Issues]
- **Ask Questions**: [Community Forum]
- **Email**: support@pbc-dashboard.example
- **Documentation**: This file!

Thank you for using PBC Visual Classification Dashboard! 🙏