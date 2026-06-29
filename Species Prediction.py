"""
Iris Species Predictor — Interactive Input + Visualization
===========================================================
Enter sepal & petal measurements → get predicted species + charts

Usage:
    python iris_predictor.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

CLASSES   = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
COLORS    = ['#2a78d6', '#1baf7a', '#eda100']
ICONS     = ['🌸', '🌿', '🌺']
FEAT_COLS = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
FEAT_UNITS = ['cm', 'cm', 'cm', 'cm']
RANGES    = [(4.0, 8.0), (2.0, 4.5), (1.0, 7.0), (0.1, 2.5)]

IRIS_DATA = [
    [5.1,3.5,1.4,0.2,'Iris-setosa'],[4.9,3.0,1.4,0.2,'Iris-setosa'],[4.7,3.2,1.3,0.2,'Iris-setosa'],
    [4.6,3.1,1.5,0.2,'Iris-setosa'],[5.0,3.6,1.4,0.2,'Iris-setosa'],[5.4,3.9,1.7,0.4,'Iris-setosa'],
    [4.6,3.4,1.4,0.3,'Iris-setosa'],[5.0,3.4,1.5,0.2,'Iris-setosa'],[4.4,2.9,1.4,0.2,'Iris-setosa'],
    [4.9,3.1,1.5,0.1,'Iris-setosa'],[5.4,3.7,1.5,0.2,'Iris-setosa'],[4.8,3.4,1.6,0.2,'Iris-setosa'],
    [4.8,3.0,1.4,0.1,'Iris-setosa'],[4.3,3.0,1.1,0.1,'Iris-setosa'],[5.8,4.0,1.2,0.2,'Iris-setosa'],
    [5.7,4.4,1.5,0.4,'Iris-setosa'],[5.4,3.9,1.3,0.4,'Iris-setosa'],[5.1,3.5,1.4,0.3,'Iris-setosa'],
    [5.7,3.8,1.7,0.3,'Iris-setosa'],[5.1,3.8,1.5,0.3,'Iris-setosa'],[5.4,3.4,1.7,0.2,'Iris-setosa'],
    [5.1,3.7,1.5,0.4,'Iris-setosa'],[4.6,3.6,1.0,0.2,'Iris-setosa'],[5.1,3.3,1.7,0.5,'Iris-setosa'],
    [4.8,3.4,1.9,0.2,'Iris-setosa'],[5.0,3.0,1.6,0.2,'Iris-setosa'],[5.0,3.4,1.6,0.4,'Iris-setosa'],
    [5.2,3.5,1.5,0.2,'Iris-setosa'],[5.2,3.4,1.4,0.2,'Iris-setosa'],[4.7,3.2,1.6,0.2,'Iris-setosa'],
    [4.8,3.1,1.6,0.2,'Iris-setosa'],[5.4,3.4,1.5,0.4,'Iris-setosa'],[5.2,4.1,1.5,0.1,'Iris-setosa'],
    [5.5,4.2,1.4,0.2,'Iris-setosa'],[4.9,3.1,1.5,0.2,'Iris-setosa'],[5.0,3.2,1.2,0.2,'Iris-setosa'],
    [5.5,3.5,1.3,0.2,'Iris-setosa'],[4.9,3.6,1.4,0.1,'Iris-setosa'],[4.4,3.0,1.3,0.2,'Iris-setosa'],
    [5.1,3.4,1.5,0.2,'Iris-setosa'],[5.0,3.5,1.3,0.3,'Iris-setosa'],[4.5,2.3,1.3,0.3,'Iris-setosa'],
    [4.4,3.2,1.3,0.2,'Iris-setosa'],[5.0,3.5,1.6,0.6,'Iris-setosa'],[5.1,3.8,1.9,0.4,'Iris-setosa'],
    [4.8,3.0,1.4,0.3,'Iris-setosa'],[5.1,3.8,1.6,0.2,'Iris-setosa'],[4.6,3.2,1.4,0.2,'Iris-setosa'],
    [5.3,3.7,1.5,0.2,'Iris-setosa'],[5.0,3.3,1.4,0.2,'Iris-setosa'],
    [7.0,3.2,4.7,1.4,'Iris-versicolor'],[6.4,3.2,4.5,1.5,'Iris-versicolor'],[6.9,3.1,4.9,1.5,'Iris-versicolor'],
    [5.5,2.3,4.0,1.3,'Iris-versicolor'],[6.5,2.8,4.6,1.5,'Iris-versicolor'],[5.7,2.8,4.5,1.3,'Iris-versicolor'],
    [6.3,3.3,4.7,1.6,'Iris-versicolor'],[4.9,2.4,3.3,1.0,'Iris-versicolor'],[6.6,2.9,4.6,1.3,'Iris-versicolor'],
    [5.2,2.7,3.9,1.4,'Iris-versicolor'],[5.0,2.0,3.5,1.0,'Iris-versicolor'],[5.9,3.0,4.2,1.5,'Iris-versicolor'],
    [6.0,2.2,4.0,1.0,'Iris-versicolor'],[6.1,2.9,4.7,1.4,'Iris-versicolor'],[5.6,2.9,3.6,1.3,'Iris-versicolor'],
    [6.7,3.1,4.4,1.4,'Iris-versicolor'],[5.6,3.0,4.5,1.5,'Iris-versicolor'],[5.8,2.7,4.1,1.0,'Iris-versicolor'],
    [6.2,2.2,4.5,1.5,'Iris-versicolor'],[5.6,2.5,3.9,1.1,'Iris-versicolor'],[5.9,3.2,4.8,1.8,'Iris-versicolor'],
    [6.1,2.8,4.0,1.3,'Iris-versicolor'],[6.3,2.5,4.9,1.5,'Iris-versicolor'],[6.1,2.8,4.7,1.2,'Iris-versicolor'],
    [6.4,2.9,4.3,1.3,'Iris-versicolor'],[6.6,3.0,4.4,1.4,'Iris-versicolor'],[6.8,2.8,4.8,1.4,'Iris-versicolor'],
    [6.7,3.0,5.0,1.7,'Iris-versicolor'],[6.0,2.9,4.5,1.5,'Iris-versicolor'],[5.7,2.6,3.5,1.0,'Iris-versicolor'],
    [5.5,2.4,3.8,1.1,'Iris-versicolor'],[5.5,2.4,3.7,1.0,'Iris-versicolor'],[5.8,2.7,3.9,1.2,'Iris-versicolor'],
    [6.0,2.7,5.1,1.6,'Iris-versicolor'],[5.4,3.0,4.5,1.5,'Iris-versicolor'],[6.0,3.4,4.5,1.6,'Iris-versicolor'],
    [6.7,3.1,4.7,1.5,'Iris-versicolor'],[6.3,2.3,4.4,1.3,'Iris-versicolor'],[5.6,3.0,4.1,1.3,'Iris-versicolor'],
    [5.5,2.5,4.0,1.3,'Iris-versicolor'],[5.5,2.6,4.4,1.2,'Iris-versicolor'],[6.1,3.0,4.6,1.4,'Iris-versicolor'],
    [5.8,2.6,4.0,1.2,'Iris-versicolor'],[5.0,2.3,3.3,1.0,'Iris-versicolor'],[5.6,2.7,4.2,1.3,'Iris-versicolor'],
    [5.7,3.0,4.2,1.2,'Iris-versicolor'],[5.7,2.9,4.2,1.3,'Iris-versicolor'],[6.2,2.9,4.3,1.3,'Iris-versicolor'],
    [5.1,2.5,3.0,1.1,'Iris-versicolor'],[5.7,2.8,4.1,1.3,'Iris-versicolor'],
    [6.3,3.3,6.0,2.5,'Iris-virginica'],[5.8,2.7,5.1,1.9,'Iris-virginica'],[7.1,3.0,5.9,2.1,'Iris-virginica'],
    [6.3,2.9,5.6,1.8,'Iris-virginica'],[6.5,3.0,5.8,2.2,'Iris-virginica'],[7.6,3.0,6.6,2.1,'Iris-virginica'],
    [4.9,2.5,4.5,1.7,'Iris-virginica'],[7.3,2.9,6.3,1.8,'Iris-virginica'],[6.7,2.5,5.8,1.8,'Iris-virginica'],
    [7.2,3.6,6.1,2.5,'Iris-virginica'],[6.5,3.2,5.1,2.0,'Iris-virginica'],[6.4,2.7,5.3,1.9,'Iris-virginica'],
    [6.8,3.0,5.5,2.1,'Iris-virginica'],[5.7,2.5,5.0,2.0,'Iris-virginica'],[5.8,2.8,5.1,2.4,'Iris-virginica'],
    [6.4,3.2,5.3,2.3,'Iris-virginica'],[6.5,3.0,5.5,1.8,'Iris-virginica'],[7.7,3.8,6.7,2.2,'Iris-virginica'],
    [7.7,2.6,6.9,2.3,'Iris-virginica'],[6.0,2.2,5.0,1.5,'Iris-virginica'],[6.9,3.2,5.7,2.3,'Iris-virginica'],
    [5.6,2.8,4.9,2.0,'Iris-virginica'],[7.7,2.8,6.7,2.0,'Iris-virginica'],[6.3,2.7,4.9,1.8,'Iris-virginica'],
    [6.7,3.3,5.7,2.1,'Iris-virginica'],[7.2,3.2,6.0,1.8,'Iris-virginica'],[6.2,2.8,4.8,1.8,'Iris-virginica'],
    [6.1,3.0,4.9,1.8,'Iris-virginica'],[6.4,2.8,5.6,2.1,'Iris-virginica'],[7.2,3.0,5.8,1.6,'Iris-virginica'],
    [7.4,2.8,6.1,1.9,'Iris-virginica'],[7.9,3.8,6.4,2.0,'Iris-virginica'],[6.4,2.8,5.6,2.2,'Iris-virginica'],
    [6.3,2.8,5.1,1.5,'Iris-virginica'],[6.1,2.6,5.6,1.4,'Iris-virginica'],[7.7,3.0,6.1,2.3,'Iris-virginica'],
    [6.3,3.4,5.6,2.4,'Iris-virginica'],[6.4,3.1,5.5,1.8,'Iris-virginica'],[6.0,3.0,4.8,1.8,'Iris-virginica'],
    [6.9,3.1,5.4,2.1,'Iris-virginica'],[6.7,3.1,5.6,2.4,'Iris-virginica'],[6.9,3.1,5.1,2.3,'Iris-virginica'],
    [5.8,2.7,5.1,1.9,'Iris-virginica'],[6.8,3.2,5.9,2.3,'Iris-virginica'],[6.7,3.3,5.7,2.5,'Iris-virginica'],
    [6.7,3.0,5.2,2.3,'Iris-virginica'],[6.3,2.5,5.0,1.9,'Iris-virginica'],[6.5,3.0,5.2,2.0,'Iris-virginica'],
    [6.2,3.4,5.4,2.3,'Iris-virginica'],[5.9,3.0,5.1,1.8,'Iris-virginica'],
]

def compute_stats():
    """Compute mean and variance for each class and feature."""
    means, variances = {}, {}
    for cls in CLASSES:
        rows = [r[:4] for r in IRIS_DATA if r[4] == cls]
        arr  = np.array(rows)
        means[cls]     = arr.mean(axis=0)
        variances[cls] = np.maximum(arr.var(axis=0), 1e-9)
    return means, variances

MEANS, VARS = compute_stats()

def gaussian_pdf(x, mu, sig2):
    return (1.0 / np.sqrt(2 * np.pi * sig2)) * np.exp(-((x - mu) ** 2) / (2 * sig2))

def predict(features):
    """Gaussian Naive Bayes prediction."""
    log_scores = {}
    for cls in CLASSES:
        lp = np.log(1.0 / 3)
        for i, x in enumerate(features):
            lp += np.log(gaussian_pdf(x, MEANS[cls][i], VARS[cls][i]) + 1e-300)
        log_scores[cls] = lp

    max_s = max(log_scores.values())
    exps  = {c: np.exp(log_scores[c] - max_s) for c in CLASSES}
    total = sum(exps.values())
    probs = {c: exps[c] / total for c in CLASSES}
    best  = max(probs, key=probs.get)
    return best, probs

def get_input():
    """Prompt user to enter four measurements with validation."""
    prompts = [
        ("Sepal length", 4.0, 8.0),
        ("Sepal width",  2.0, 4.5),
        ("Petal length", 1.0, 7.0),
        ("Petal width",  0.1, 2.5),
    ]
    values = []
    print("\n" + "=" * 50)
    print("  IRIS SPECIES PREDICTOR")
    print("=" * 50)
    print("  Enter measurements (in cm) to classify.\n")

    for name, lo, hi in prompts:
        while True:
            try:
                val = float(input(f"  {name} ({lo}–{hi} cm): "))
                if lo <= val <= hi:
                    values.append(val)
                    break
                else:
                    print(f"  ✗  Value out of range. Must be between {lo} and {hi}.")
            except ValueError:
                print("  ✗  Please enter a number.")

    return values

def visualize(features, predicted, probs):
    """Create a 4-panel visualization."""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#fafaf9')
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    pred_idx = CLASSES.index(predicted)
    pred_col = COLORS[pred_idx]

    # ── Panel 1: Probability bar chart ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#fafaf9')
    class_names = [c.replace('Iris-', '') for c in CLASSES]
    pct_vals    = [probs[c] * 100 for c in CLASSES]
    bars = ax1.barh(class_names, pct_vals, color=COLORS, height=0.5, edgecolor='none')
    for bar, pct in zip(bars, pct_vals):
        ax1.text(min(pct + 1.5, 97), bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va='center', ha='left', fontsize=11, fontweight='500',
                 color='#2c2c2a')
    ax1.set_xlim(0, 108)
    ax1.set_xlabel('Probability (%)', fontsize=10, color='#898781')
    ax1.set_title('Prediction probabilities', fontsize=12, fontweight='500',
                  color='#0b0b0b', pad=10)
    ax1.tick_params(colors='#898781', labelsize=10)
    ax1.spines[['top','right','bottom']].set_visible(False)
    ax1.spines['left'].set_color('#e1e0d9')
    ax1.xaxis.grid(True, color='#e1e0d9', linewidth=0.7)
    ax1.set_axisbelow(True)

    # ── Panel 2: Radar / spider chart ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    ax2.set_facecolor('#fafaf9')
    n_feat   = len(FEAT_COLS)
    angles   = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles  += angles[:1]
    max_vals = [r[1] for r in RANGES]

    for cls, col in zip(CLASSES, COLORS):
        vals  = (MEANS[cls] / max_vals * 10).tolist()
        vals += vals[:1]
        ax2.plot(angles, vals, color=col, linewidth=1.5)
        ax2.fill(angles, vals, color=col, alpha=0.08)

    user_norm  = (np.array(features) / max_vals * 10).tolist()
    user_norm += user_norm[:1]
    ax2.plot(angles, user_norm, color='#e34948', linewidth=2.5, linestyle='--')
    ax2.fill(angles, user_norm, color='#e34948', alpha=0.12)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(FEAT_COLS, fontsize=9, color='#5f5e5a')
    ax2.set_yticklabels([])
    ax2.set_title('Feature comparison\nvs species means', fontsize=12,
                  fontweight='500', color='#0b0b0b', pad=18)
    legend_patches = [mpatches.Patch(color=c, label=n.replace('Iris-',''))
                      for c, n in zip(COLORS, CLASSES)]
    legend_patches.append(mpatches.Patch(color='#e34948', label='Your input'))
    ax2.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1.35, -0.15),
               fontsize=9, frameon=False)

    # ── Panel 3: Feature value bars vs class means ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#fafaf9')
    x     = np.arange(n_feat)
    width = 0.2

    for i, (cls, col) in enumerate(zip(CLASSES, COLORS)):
        ax3.bar(x + i * width, MEANS[cls], width, color=col, alpha=0.75,
                label=cls.replace('Iris-',''), edgecolor='none')

    ax3.bar(x + 3 * width, features, width, color='#e34948', alpha=0.9,
            label='Your input', edgecolor='none')
    ax3.set_xticks(x + 1.5 * width)
    ax3.set_xticklabels(FEAT_COLS, fontsize=9, color='#5f5e5a')
    ax3.set_ylabel('Value (cm)', fontsize=10, color='#898781')
    ax3.set_title('Your values vs class means', fontsize=12, fontweight='500',
                  color='#0b0b0b', pad=10)
    ax3.tick_params(colors='#898781', labelsize=9)
    ax3.spines[['top','right']].set_visible(False)
    ax3.spines[['left','bottom']].set_color('#e1e0d9')
    ax3.yaxis.grid(True, color='#e1e0d9', linewidth=0.7)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, frameon=False, ncol=2, loc='upper left')

    # ── Panel 4: Result summary ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(pred_col + '18')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    ax4.text(0.5, 0.88, 'Predicted species', ha='center', va='center',
             fontsize=11, color='#5f5e5a')
    ax4.text(0.5, 0.72, ICONS[pred_idx], ha='center', va='center', fontsize=40)
    ax4.text(0.5, 0.55, predicted, ha='center', va='center',
             fontsize=16, fontweight='500', color=pred_col)
    ax4.text(0.5, 0.43, f"Confidence: {probs[predicted]*100:.1f}%",
             ha='center', va='center', fontsize=12, color='#5f5e5a')

    ax4.add_patch(mpatches.FancyBboxPatch((0.05, 0.08), 0.9, 0.26,
        boxstyle='round,pad=0.02', linewidth=0, facecolor='white', alpha=0.6))

    feat_summary = (
        f"Sepal: {features[0]:.1f} × {features[1]:.1f} cm\n"
        f"Petal: {features[2]:.1f} × {features[3]:.1f} cm"
    )
    ax4.text(0.5, 0.21, feat_summary, ha='center', va='center',
             fontsize=11, color='#2c2c2a', linespacing=1.8)

    fig.suptitle('Iris Species Classifier — Gaussian Naive Bayes',
                 fontsize=14, fontweight='500', color='#0b0b0b', y=0.98)

    plt.savefig('iris_prediction.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("\n  Chart saved → iris_prediction.png")
    plt.show()

def main():
    features  = get_input()
    predicted, probs = predict(features)

    print("\n" + "─" * 50)
    print(f"  Predicted species : {predicted}  {ICONS[CLASSES.index(predicted)]}")
    print(f"  Confidence        : {probs[predicted]*100:.1f}%")
    print("─" * 50)
    print("  Class probabilities:")
    for cls, col in zip(CLASSES, ["🔵","🟢","🟡"]):
        bar = "█" * int(probs[cls] * 30)
        print(f"  {col}  {cls:<20} {bar:<30} {probs[cls]*100:5.1f}%")
    print("─" * 50)
    print("\n  Generating visualisation...")

    visualize(features, predicted, probs)

    again = input("\n  Predict another flower? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("\n  Done. Goodbye!\n")

if __name__ == "__main__":
    main()