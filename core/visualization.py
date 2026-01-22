import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 



def visualize_features(df, plot_features):

    plot_df = df[plot_features].copy()

    class_names = {0: "Beam", 1: "Column", 2: "Wall", 3: "Slab"}
    plot_df['Class'] = plot_df['label'].map(class_names)

    # 3. Create the Pairplot
    sns.set_theme(style="ticks")
    g = sns.pairplot(plot_df.drop('label', axis=1), hue="Class", palette="bright", corner=True)
    g.fig.suptitle("Geometric Separability of Construction Elements", y=1.02)
    #g.set(xscale="log")

    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Visualizes which features the Random Forest model prioritized.
    Addresses Requirement 2.2: Documentation of model/heuristic tuning.
    """
    # 1. Extract importances from the model
    importances = model.feature_importances_
    
    # 2. Create a DataFrame for easy plotting
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # 3. Sort features by importance
    feat_df = feat_df.sort_values(by='Importance', ascending=False)

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Use a gradient palette to emphasize the top features
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feat_df, 
        palette='magma', 
        hue='Feature', 
        legend=False
    )
    
    plt.title("BIM Classifier: Feature Importance", fontsize=14, fontweight='bold')
    plt.xlabel("Gini Importance (Weight)", fontsize=12)
    plt.ylabel("Geometric Feature", fontsize=12)
    
    plt.tight_layout()
    plt.show()

    