import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory for figures
os.makedirs('project_figures', exist_ok=True)

# Figure 1: Traditional Approach to Drug Risk Assessment
plt.figure(figsize=(10, 6))
plt.plot([0, 1, 2, 3, 4], [0, 1, 0, 2, 1], 'bo-', linewidth=2)
plt.title('Traditional Approach to Drug Risk Assessment', fontsize=16)
plt.xlabel('Assessment Steps', fontsize=14)
plt.ylabel('Complexity Level', fontsize=14)
plt.xticks([0, 1, 2, 3, 4], ['Initial Data\nCollection', 'Binary\nClassification', 'Limited\nInterpretation', 'No Data\nPersistence', 'Static\nOutput'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(-0.5, 2.5)
plt.text(2, 2.1, 'Research-Oriented\nApproach', fontsize=12, 
         bbox=dict(facecolor='yellow', alpha=0.2), ha='center')
plt.savefig('project_figures/Figure1_Traditional_Approach.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: System Components and Data Flow
plt.figure(figsize=(12, 8))
components = ['User\nInterface', 'Data\nPreprocessing', 'Model\nPrediction', 'Risk\nAssessment', 'Database\nStorage', 'Statistical\nAnalysis']
pos = np.arange(len(components))
plt.plot(pos, [3, 2, 1, 2, 3, 2], 'ro-', markersize=15, linewidth=3)

# Add arrows to indicate flow
for i in range(len(components)-1):
    plt.annotate('', xy=(pos[i+1], 2), xytext=(pos[i], 2),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=10))

plt.title('System Components and Data Flow', fontsize=16)
plt.xticks(pos, components, fontsize=12)
plt.yticks([])
plt.grid(False)
plt.ylim(0, 4)

# Add descriptions
descriptions = [
    'Streamlit Web App',
    'Feature Engineering',
    'ML Models',
    'Risk Algorithm',
    'PostgreSQL',
    'Visualization'
]

for i, desc in enumerate(descriptions):
    plt.text(pos[i], 0.5, desc, ha='center', fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.3))

plt.savefig('project_figures/Figure2_System_Components.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Architecture Diagram
plt.figure(figsize=(12, 9))

# Define the layers
layers = ['User Interface Layer', 'Application Logic Layer', 'Prediction Engine', 'Data Persistence Layer', 'Statistical Analysis Module']
positions = [5, 4, 3, 2, 1]

# Plot rectangles for each layer
for i, (layer, pos) in enumerate(zip(layers, positions)):
    rect = plt.Rectangle((1, pos-0.4), 8, 0.8, facecolor=f'C{i}', alpha=0.6)
    plt.gca().add_patch(rect)
    plt.text(5, pos, layer, ha='center', va='center', fontsize=14, fontweight='bold')

# Add arrows to show interactions
plt.arrow(5, 4.5, 0, 0.1, head_width=0.2, head_length=0.1, fc='black', ec='black')
plt.arrow(5, 4.5, 0, -0.1, head_width=0.2, head_length=0.1, fc='black', ec='black')

plt.arrow(5, 3.5, 0, 0.1, head_width=0.2, head_length=0.1, fc='black', ec='black')
plt.arrow(5, 3.5, 0, -0.1, head_width=0.2, head_length=0.1, fc='black', ec='black')

plt.arrow(5, 2.5, 0, 0.1, head_width=0.2, head_length=0.1, fc='black', ec='black')
plt.arrow(5, 2.5, 0, -0.1, head_width=0.2, head_length=0.1, fc='black', ec='black')

plt.arrow(7, 3, 0, -1, head_width=0.2, head_length=0.1, fc='black', ec='black')

# Add components within each layer
components = [
    ['Streamlit Forms', 'Data Visualization', 'User Input Validation'],
    ['Risk Assessment', 'Model Selection', 'Data Preprocessing'],
    ['Random Forest', 'SVM', 'XGBoost', 'Logistic Regression'],
    ['SQLAlchemy ORM', 'PostgreSQL', 'Data Retrieval'],
    ['Feature Analysis', 'Population Statistics', 'Trend Analysis']
]

for i, (comps, pos) in enumerate(zip(components, positions)):
    for j, comp in enumerate(comps):
        x_pos = 2 + j*3
        y_pos = pos - 0.2
        plt.text(x_pos, y_pos, comp, ha='center', va='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

plt.xlim(0, 10)
plt.ylim(0, 6)
plt.title('Architecture Diagram', fontsize=16)
plt.axis('off')
plt.savefig('project_figures/Figure3_Architecture_Diagram.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Machine Learning Model Training Process
plt.figure(figsize=(12, 6))

steps = ['Data\nLoading', 'Data\nPreprocessing', 'Train/Test\nSplit', 'Model\nTraining', 'Model\nEvaluation', 'Best Model\nSelection', 'Model\nSerialization']
x = np.arange(len(steps))
y = [1, 1, 1, 1, 1, 1, 1]

plt.bar(x, y, color=['royalblue', 'limegreen', 'orange', 'red', 'purple', 'brown', 'teal'])
plt.title('Machine Learning Model Training Process', fontsize=16)
plt.xticks(x, steps, fontsize=12)
plt.yticks([])

# Add arrows connecting the steps
for i in range(len(steps)-1):
    plt.annotate('', xy=(i+1, 0.5), xytext=(i, 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Add descriptions for each step
descriptions = [
    'UCI Drug Dataset',
    'Handle Missing Values\nScale Features',
    '80/20 Split\nStratified',
    'Multiple Algorithms\nHyperparameter Tuning',
    'Accuracy, Precision\nRecall, F1 Score',
    'Best F1 Score\nPer Substance',
    'Save to Disk\nwith Joblib'
]

for i, desc in enumerate(descriptions):
    plt.text(i, 0.5, desc, ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

plt.xlim(-0.5, len(steps)-0.5)
plt.ylim(0, 1.5)
plt.savefig('project_figures/Figure4_ML_Training_Process.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Risk Assessment Calculation Algorithm
plt.figure(figsize=(10, 8))

# Create flowchart with shapes
shapes = ['Input\nProbabilities', 'Calculate\nBinary Class', 'Calculate\nRisk Score', 'Apply\nThresholds', 'Generate\nExplanation', 'Identify\nSubstances of\nConcern']
positions = [(5, 7), (5, 6), (5, 5), (5, 4), (5, 3), (5, 2)]

# Add decision diamond
plt.plot([5, 6, 5, 4, 5], [4, 3, 2, 3, 4], 'k-')
plt.text(5, 3, 'Risk Level?', ha='center', va='center', fontsize=12)

# Add boxes for each step
for i, (label, pos) in enumerate(zip(shapes, positions)):
    if i != 3:  # Not the decision diamond
        rect = plt.Rectangle((pos[0]-1, pos[1]-0.4), 2, 0.8, facecolor='lightblue', alpha=0.8, ec='k')
        plt.gca().add_patch(rect)
        plt.text(pos[0], pos[1], label, ha='center', va='center', fontsize=12)

# Add arrows
for i in range(len(positions)-1):
    if i != 3:  # Skip the decision diamond connection
        plt.arrow(positions[i][0], positions[i][1]-0.4, 0, -0.2, 
                 head_width=0.2, head_length=0.1, fc='black', ec='black')

# Add outcome boxes
outcomes = ['Low Risk\n< 0.3', 'Medium Risk\n0.3-0.7', 'High Risk\n> 0.7']
outcome_pos = [(3, 3), (5, 1), (7, 3)]
colors = ['green', 'yellow', 'red']

for (label, pos, color) in zip(outcomes, outcome_pos, colors):
    rect = plt.Rectangle((pos[0]-0.8, pos[1]-0.4), 1.6, 0.8, facecolor=color, alpha=0.4, ec='k')
    plt.gca().add_patch(rect)
    plt.text(pos[0], pos[1], label, ha='center', va='center', fontsize=10)

# Connect decision to outcomes
plt.arrow(4, 3, -0.2, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
plt.arrow(5, 2, 0, -0.2, head_width=0.2, head_length=0.1, fc='black', ec='black')
plt.arrow(6, 3, 0.2, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')

plt.title('Risk Assessment Calculation Algorithm', fontsize=16)
plt.xlim(2, 8)
plt.ylim(0.5, 7.5)
plt.axis('off')
plt.savefig('project_figures/Figure5_Risk_Assessment_Algorithm.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Data-Driven Statistical Analysis Process
plt.figure(figsize=(12, 8))

# Set up hexagonal layout
labels = ['Data Collection', 'Feature Extraction', 'Statistical Testing', 
          'Pattern Identification', 'Correlation Analysis', 'Visualization']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Create hexagons with arrows
for i, (label, color) in enumerate(zip(labels, colors)):
    angle = i * (2 * np.pi / 6)
    x = 5 + 2.5 * np.cos(angle)
    y = 5 + 2.5 * np.sin(angle)
    
    # Draw hexagon
    hex_vertices = []
    for j in range(6):
        hex_angle = j * (2 * np.pi / 6)
        hex_x = x + 0.8 * np.cos(hex_angle)
        hex_y = y + 0.8 * np.sin(hex_angle)
        hex_vertices.append((hex_x, hex_y))
    
    hex_x, hex_y = zip(*hex_vertices)
    plt.fill(hex_x, hex_y, color, alpha=0.7)
    
    # Add label
    plt.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw arrow to next hexagon
    next_i = (i + 1) % 6
    next_angle = next_i * (2 * np.pi / 6)
    next_x = 5 + 2.5 * np.cos(next_angle)
    next_y = 5 + 2.5 * np.sin(next_angle)
    
    # Calculate arrow points
    mid_x = 5 + 1.7 * np.cos((angle + next_angle) / 2)
    mid_y = 5 + 1.7 * np.sin((angle + next_angle) / 2)
    
    plt.arrow(x + 0.5 * np.cos(angle + np.pi/3), 
              y + 0.5 * np.sin(angle + np.pi/3),
              (mid_x - x) * 0.6, (mid_y - y) * 0.6,
              head_width=0.2, head_length=0.2, fc='black', ec='black')

# Add central node
central_circle = plt.Circle((5, 5), 1.2, color='#17becf', alpha=0.8)
plt.gca().add_patch(central_circle)
plt.text(5, 5, 'Data-Driven\nStatistical\nAnalysis', ha='center', va='center', fontsize=12, fontweight='bold')

# Add explanatory text
explanations = [
    'User Profiles\nPrediction Results',
    'Personality Traits\nDemographic Factors',
    'ANOVA\nChi-Square\nCorrelation Tests',
    'Risk Factor\nIdentification',
    'Substance Use\nand Personality',
    'Interactive\nDashboards'
]

for i, explanation in enumerate(zip(explanations)):
    angle = i * (2 * np.pi / 6)
    x = 5 + 3.5 * np.cos(angle)
    y = 5 + 3.5 * np.sin(angle)
    plt.text(x, y, explanations[i], ha='center', va='center', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title('Data-Driven Statistical Analysis Process', fontsize=16)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.axis('off')
plt.savefig('project_figures/Figure6_Statistical_Analysis_Process.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: Proposed Approach
plt.figure(figsize=(12, 8))

# Create comparison view of traditional vs proposed approach
plt.subplot(1, 2, 1)
plt.title('Traditional Approach', fontsize=14)
x = np.arange(5)
traditional = [1, 1, 0.3, 0, 0.2]
plt.bar(x, traditional, color='gray', alpha=0.6)
plt.xticks(x, ['Binary\nClassification', 'Limited\nExplanation', 'No Data\nPersistence', 'No Risk\nLevels', 'No Statistical\nAnalysis'], rotation=45, ha='right', fontsize=10)
plt.ylim(0, 1.2)
plt.ylabel('Capability Level', fontsize=12)

plt.subplot(1, 2, 2)
plt.title('Proposed Approach', fontsize=14)
proposed = [1, 1, 1, 0.8, 1]
plt.bar(x, proposed, color='green', alpha=0.6)
plt.xticks(x, ['Multi-Model\nPrediction', 'Interpretable\nExplanation', 'Database\nPersistence', 'Graduated Risk\nLevels', 'Statistical\nAnalysis'], rotation=45, ha='right', fontsize=10)
plt.ylim(0, 1.2)

plt.suptitle('Comparison of Traditional and Proposed Approaches', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('project_figures/Figure7_Proposed_Approach.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 8: User Interface and Experience
plt.figure(figsize=(12, 8))

# Create a mockup of the UI
plt.subplot(1, 3, 1)
plt.title('User Input Form', fontsize=12)
rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor='#f0f0f0', edgecolor='k')
plt.gca().add_patch(rect)

# Add form elements
form_elements = [
    'Age Group', 'Gender', 'Education', 'Country',
    'Personality Traits', 'Impulsivity Score', 'Sensation Seeking'
]
for i, element in enumerate(form_elements):
    y_pos = 0.8 - i * 0.1
    plt.text(0.15, y_pos, element, fontsize=10)
    plt.plot([0.5, 0.8], [y_pos-0.02, y_pos-0.02], 'k-', alpha=0.3)

plt.text(0.5, 0.2, 'Submit', ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='blue', alpha=0.4, boxstyle='round,pad=0.3'))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Risk Assessment Display', fontsize=12)
rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor='#f0f0f0', edgecolor='k')
plt.gca().add_patch(rect)

# Create risk level gauges
substances = ['Cannabis', 'Alcohol', 'Nicotine', 'Ecstasy', 'Mushrooms']
risk_levels = [0.75, 0.45, 0.85, 0.25, 0.35]
colors = ['red', 'yellow', 'red', 'green', 'yellow']

for i, (substance, risk, color) in enumerate(zip(substances, risk_levels, colors)):
    y_pos = 0.8 - i * 0.15
    plt.text(0.15, y_pos, substance, fontsize=10)
    plt.Rectangle((0.4, y_pos-0.03), 0.4, 0.06, facecolor='lightgray', alpha=0.5, edgecolor='k')
    plt.gca().add_patch(plt.Rectangle((0.4, y_pos-0.03), 0.4, 0.06, facecolor='lightgray', alpha=0.5, edgecolor='k'))
    plt.gca().add_patch(plt.Rectangle((0.4, y_pos-0.03), risk*0.4, 0.06, facecolor=color, alpha=0.7, edgecolor='k'))

plt.text(0.5, 0.2, 'Download Report', ha='center', va='center', fontsize=12,
         bbox=dict(facecolor='green', alpha=0.4, boxstyle='round,pad=0.3'))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Feature Importance Visualization', fontsize=12)
rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, facecolor='#f0f0f0', edgecolor='k')
plt.gca().add_patch(rect)

# Create feature importance visualization
features = ['SS', 'Impulsive', 'Oscore', 'Age', 'Education']
importance = [0.35, 0.25, 0.20, 0.12, 0.08]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

y_pos = np.arange(len(features))
plt.barh(y_pos, importance, color=colors, alpha=0.7)
plt.yticks(y_pos, features)
plt.xlabel('Importance')
plt.xlim(0, 0.4)
plt.gca().invert_yaxis()  # labels read top-to-bottom

plt.tight_layout()
plt.savefig('project_figures/Figure8_User_Interface.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 9: Database Schema for User Profile and Prediction Storage
plt.figure(figsize=(12, 8))

# Create database schema visualization
tables = [
    ('user_profiles', ['id (PK)', 'created_at', 'age_group', 'gender', 'education', 'country', 'ethnicity', 
                      'nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'sensation_seeking']),
    ('prediction_results', ['id (PK)', 'user_id (FK)', 'created_at', 'substance', 'prediction', 
                           'probability', 'model_type', 'risk_level', 'risk_factors'])
]

# Draw tables
for i, (table, fields) in enumerate(tables):
    x = 3 + i * 6
    y = 4
    
    # Draw table body
    table_height = len(fields) * 0.4 + 0.8
    rect = plt.Rectangle((x-2.5, y-table_height/2), 5, table_height, facecolor='#b3d1ff', edgecolor='k')
    plt.gca().add_patch(rect)
    
    # Draw table header
    header = plt.Rectangle((x-2.5, y+table_height/2-0.8), 5, 0.8, facecolor='#004080', edgecolor='k')
    plt.gca().add_patch(header)
    plt.text(x, y+table_height/2-0.4, table, color='white', fontsize=12, fontweight='bold', ha='center', va='center')
    
    # Draw fields
    for j, field in enumerate(fields):
        field_y = y + table_height/2 - 1.0 - j * 0.4
        plt.text(x, field_y, field, fontsize=10, ha='center', va='center')
        if j == 0:  # Primary key
            plt.plot([x-2.4, x+2.4], [field_y+0.15, field_y+0.15], 'k-', alpha=0.3)

# Draw relationship
plt.arrow(8.6, 4 - 0.4, 1.8, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
plt.text(9.5, 4-0.1, '1:N', fontsize=12, ha='center', va='bottom')

plt.title('Database Schema for User Profile and Prediction Storage', fontsize=16)
plt.xlim(0, 12)
plt.ylim(0, 8)
plt.axis('off')
plt.savefig('project_figures/Figure9_Database_Schema.png', dpi=300, bbox_inches='tight')
plt.close()

# Table 1: User Stories (already in Markdown format in the report)

# Table 2: Comparison Analysis of Model Performance (already in Markdown format in the report)

print("All figures have been generated and saved to the 'project_figures' directory.")