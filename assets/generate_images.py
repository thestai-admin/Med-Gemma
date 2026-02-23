"""Generate writeup images for MedGemma Impact Challenge submission."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──
BG = '#0f172a'        # dark navy
CARD_BG = '#1e293b'   # card background
ACCENT = '#38bdf8'    # sky blue
ACCENT2 = '#818cf8'   # indigo
ACCENT3 = '#34d399'   # emerald
ACCENT4 = '#f472b6'   # pink
ACCENT5 = '#fb923c'   # orange
WHITE = '#f8fafc'
GRAY = '#94a3b8'
DARK_CARD = '#0f172a'


def rounded_box(ax, x, y, w, h, label, sublabel=None, color=ACCENT, fontsize=10, sublabel_size=7):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color + '22',
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.015, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=WHITE, family='sans-serif')
        ax.text(x + w/2, y + h/2 - 0.025, sublabel, ha='center', va='center',
                fontsize=sublabel_size, color=GRAY, family='sans-serif')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=WHITE, family='sans-serif')


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle='arc3,rad=0'))


# ════════════════════════════════════════════════════════════════
# IMAGE 1: Card / Thumbnail (560 x 280)
# ════════════════════════════════════════════════════════════════
def create_card_thumbnail():
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), dpi=80)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.92, 'PrimaCare AI', ha='center', va='center',
            fontsize=22, fontweight='bold', color=WHITE, family='sans-serif')
    ax.text(0.5, 0.82, 'Multi-Agent Chest X-Ray Diagnostic System',
            ha='center', va='center', fontsize=10, color=GRAY, family='sans-serif')

    # Pipeline boxes
    agents = [
        ('Intake', 'HPI', ACCENT),
        ('Imaging', 'CXR', ACCENT2),
        ('Reasoning', 'Dx', ACCENT3),
        ('Guidelines', 'RAG', ACCENT4),
        ('Education', 'Lit.', ACCENT5),
    ]

    box_w = 0.14
    box_h = 0.18
    gap = 0.035
    total_w = len(agents) * box_w + (len(agents) - 1) * gap
    start_x = (1 - total_w) / 2
    y = 0.42

    # Patient icon (left)
    ax.text(start_x - 0.07, y + box_h/2, '\u2695', ha='center', va='center',
            fontsize=20, color=ACCENT, family='sans-serif')
    ax.text(start_x - 0.07, y + box_h/2 - 0.08, 'Patient', ha='center', va='center',
            fontsize=7, color=GRAY, family='sans-serif')

    # Arrow from patient to first box
    arrow(ax, start_x - 0.03, y + box_h/2, start_x + 0.01, y + box_h/2, GRAY)

    for i, (name, sub, color) in enumerate(agents):
        x = start_x + i * (box_w + gap)
        rounded_box(ax, x, y, box_w, box_h, name, sub, color, fontsize=9, sublabel_size=7)

        # Arrow to next
        if i < len(agents) - 1:
            arrow(ax, x + box_w + 0.005, y + box_h/2,
                  x + box_w + gap - 0.005, y + box_h/2, GRAY)

    # Arrow from last box to Report
    last_x = start_x + (len(agents) - 1) * (box_w + gap)
    arrow(ax, last_x + box_w + 0.005, y + box_h/2,
          last_x + box_w + 0.05, y + box_h/2, GRAY)
    ax.text(last_x + box_w + 0.08, y + box_h/2, '\u2611', ha='center', va='center',
            fontsize=18, color=ACCENT3, family='sans-serif')
    ax.text(last_x + box_w + 0.08, y + box_h/2 - 0.08, 'Report', ha='center', va='center',
            fontsize=7, color=GRAY, family='sans-serif')

    # Bottom: model badges
    badges = [
        ('MedGemma 1.5 4B', ACCENT),
        ('MedSigLIP 448', ACCENT2),
        ('ONNX INT8 Edge', ACCENT5),
    ]
    badge_y = 0.12
    total_badge_w = 0.7
    badge_w = 0.2
    badge_gap = (total_badge_w - len(badges) * badge_w) / (len(badges) - 1)
    badge_start = (1 - total_badge_w) / 2

    for i, (label, color) in enumerate(badges):
        bx = badge_start + i * (badge_w + badge_gap)
        box = FancyBboxPatch(
            (bx, badge_y - 0.03), badge_w, 0.07,
            boxstyle="round,pad=0.01",
            facecolor=color + '33',
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(bx + badge_w/2, badge_y + 0.005, label, ha='center', va='center',
                fontsize=8, color=WHITE, family='sans-serif')

    # Tracks label
    ax.text(0.5, 0.03, 'Main Track  |  Agentic Workflow  |  Novel Task  |  Edge AI',
            ha='center', va='center', fontsize=7, color=GRAY, family='sans-serif')

    plt.tight_layout(pad=0.3)
    fig.savefig('assets/card_thumbnail.png', dpi=80, facecolor=BG,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Saved: assets/card_thumbnail.png')


# ════════════════════════════════════════════════════════════════
# IMAGE 2: Architecture Diagram (media gallery)
# ════════════════════════════════════════════════════════════════
def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'PrimaCare AI  -  System Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=WHITE, family='sans-serif')

    # ── Cloud pipeline (top) ──
    cloud_y = 0.72
    ax.text(0.08, cloud_y + 0.12, 'CLOUD (GPU)', fontsize=9, fontweight='bold',
            color=ACCENT, family='sans-serif')

    # Cloud background
    cloud_box = FancyBboxPatch(
        (0.04, cloud_y - 0.07), 0.92, 0.22,
        boxstyle="round,pad=0.01",
        facecolor=ACCENT + '0a',
        edgecolor=ACCENT + '44',
        linewidth=1, linestyle='--',
    )
    ax.add_patch(cloud_box)

    agents = [
        ('1. Intake\nAgent', 'Structured\nHPI', ACCENT),
        ('2. Imaging\nAgent', 'CXR Analysis\n+ MedSigLIP', ACCENT2),
        ('3. Reasoning\nAgent', 'Differential\nDx + Workup', ACCENT3),
        ('4. Guidelines\nAgent', 'RAG-based\nEvidence', ACCENT4),
        ('5. Education\nAgent', '3 Reading\nLevels', ACCENT5),
    ]

    box_w = 0.145
    box_h = 0.14
    gap = 0.025
    total = len(agents) * box_w + (len(agents) - 1) * gap
    sx = (1 - total) / 2

    for i, (name, sub, color) in enumerate(agents):
        x = sx + i * (box_w + gap)
        rounded_box(ax, x, cloud_y - 0.04, box_w, box_h, name, sub, color, fontsize=8, sublabel_size=6)
        if i < len(agents) - 1:
            arrow(ax, x + box_w + 0.003, cloud_y + box_h/2 - 0.04,
                  x + box_w + gap - 0.003, cloud_y + box_h/2 - 0.04, GRAY)

    # ── Edge tier (bottom left) ──
    edge_y = 0.28
    ax.text(0.08, edge_y + 0.18, 'EDGE (CPU Only)', fontsize=9, fontweight='bold',
            color=ACCENT5, family='sans-serif')

    edge_box = FancyBboxPatch(
        (0.04, edge_y - 0.03), 0.42, 0.22,
        boxstyle="round,pad=0.01",
        facecolor=ACCENT5 + '0a',
        edgecolor=ACCENT5 + '44',
        linewidth=1, linestyle='--',
    )
    ax.add_patch(edge_box)

    rounded_box(ax, 0.07, edge_y, 0.16, 0.12, 'MedSigLIP\nONNX INT8', '422 MB', ACCENT5, fontsize=8, sublabel_size=6)
    rounded_box(ax, 0.27, edge_y, 0.16, 0.12, 'Binary\nScreening', 'Pneumonia?', ACCENT5, fontsize=8, sublabel_size=6)
    arrow(ax, 0.235, edge_y + 0.06, 0.265, edge_y + 0.06, GRAY)

    # Arrow from edge screening to cloud (if positive)
    ax.annotate('', xy=(0.5, cloud_y - 0.07), xytext=(0.39, edge_y + 0.12),
                arrowprops=dict(arrowstyle='->', color=ACCENT3, lw=1.5,
                                connectionstyle='arc3,rad=-0.2'))
    ax.text(0.42, 0.5, 'Positive\nEscalate', fontsize=7, color=ACCENT3,
            ha='center', family='sans-serif')

    # ── Metrics (bottom right) ──
    metrics_x = 0.54
    ax.text(metrics_x, edge_y + 0.18, 'KEY METRICS', fontsize=9, fontweight='bold',
            color=ACCENT3, family='sans-serif')

    metrics_box = FancyBboxPatch(
        (metrics_x - 0.02, edge_y - 0.03), 0.46, 0.22,
        boxstyle="round,pad=0.01",
        facecolor=ACCENT3 + '0a',
        edgecolor=ACCENT3 + '44',
        linewidth=1, linestyle='--',
    )
    ax.add_patch(metrics_box)

    metrics = [
        'F1: 0.73  |  Recall: 1.0  |  100 samples',
        'Pipeline: ~155s (GPU)  |  Edge: ~4s (CPU)',
        'Model: 1635 MB (FP32) -> 422 MB (INT8)',
        '42 tests passing  |  Bootstrap 95% CI',
    ]
    for j, line in enumerate(metrics):
        ax.text(metrics_x + 0.22, edge_y + 0.12 - j * 0.04, line,
                ha='center', va='center', fontsize=7, color=WHITE, family='sans-serif')

    # ── Models bar (bottom) ──
    ax.text(0.5, 0.08, 'MedGemma 1.5 4B  +  MedSigLIP 448  +  sentence-transformers  +  ONNX Runtime',
            ha='center', va='center', fontsize=8, color=GRAY, family='sans-serif')

    # Input arrow (left of cloud)
    ax.text(0.02, cloud_y + 0.03, 'CXR\n+\nHistory', ha='center', va='center',
            fontsize=7, color=GRAY, family='sans-serif')

    # Output arrow (right of cloud)
    right_x = sx + (len(agents) - 1) * (box_w + gap) + box_w + 0.02
    ax.text(0.98, cloud_y + 0.03, 'Clinical\nReport\n+\nEducation', ha='center', va='center',
            fontsize=7, color=GRAY, family='sans-serif')

    plt.tight_layout(pad=0.3)
    fig.savefig('assets/architecture_diagram.png', dpi=120, facecolor=BG,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Saved: assets/architecture_diagram.png')


# ════════════════════════════════════════════════════════════════
# IMAGE 3: Evaluation Results (media gallery)
# ════════════════════════════════════════════════════════════════
def create_eval_image():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=120)
    fig.patch.set_facecolor(BG)

    # ── Left: Confusion-style metrics ──
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.set_title('Binary Pneumonia Classification', fontsize=11,
                  fontweight='bold', color=WHITE, pad=12, family='sans-serif')

    categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']
    values = [0.63, 0.575, 1.0, 0.26, 0.73]
    colors_bar = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5]

    bars = ax.barh(categories, values, color=[c + '88' for c in colors_bar],
                   edgecolor=colors_bar, linewidth=1.5, height=0.6)

    ax.set_xlim(0, 1.15)
    ax.tick_params(colors=GRAY, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRAY + '44')
    ax.spines['left'].set_color(GRAY + '44')
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8, color=WHITE, family='sans-serif')

    ax.text(0.5, -0.12, '100 balanced samples  |  Threshold: 0.30  |  F1 95% CI: [0.64, 0.80]',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=7, color=GRAY, family='sans-serif')

    # ── Right: Edge vs GPU comparison ──
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.set_title('GPU vs Edge Comparison', fontsize=11,
                   fontweight='bold', color=WHITE, pad=12, family='sans-serif')

    metrics_labels = ['Size (MB)', 'Latency (s)', 'F1', 'Recall']
    gpu_vals = [3500, 17.0, 0.73, 1.0]
    edge_vals = [422, 4.1, 0.0, 0.0]  # edge currently broken, will update

    x_pos = np.arange(len(metrics_labels))
    w = 0.35

    # Normalize for display (different scales)
    gpu_norm = [v / max(g, e, 1) for v, g, e in zip(gpu_vals, gpu_vals, edge_vals)]
    edge_norm = [v / max(g, e, 1) for v, g, e in zip(edge_vals, gpu_vals, edge_vals)]

    bars1 = ax2.bar(x_pos - w/2, gpu_norm, w, label='GPU (Cloud)',
                     color=ACCENT + '88', edgecolor=ACCENT, linewidth=1.5)
    bars2 = ax2.bar(x_pos + w/2, edge_norm, w, label='Edge (CPU)',
                     color=ACCENT5 + '88', edgecolor=ACCENT5, linewidth=1.5)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_labels, fontsize=8, color=GRAY)
    ax2.set_yticks([])
    ax2.tick_params(colors=GRAY)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_color(GRAY + '44')
    ax2.spines['left'].set_visible(False)

    # Add value labels
    for bar, val in zip(bars1, gpu_vals):
        label = f'{val:.0f}' if val >= 10 else f'{val:.2f}'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 label, ha='center', va='bottom', fontsize=7, color=ACCENT, family='sans-serif')
    for bar, val in zip(bars2, edge_vals):
        label = f'{val:.0f}' if val >= 10 else f'{val:.1f}'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 label, ha='center', va='bottom', fontsize=7, color=ACCENT5, family='sans-serif')

    ax2.legend(fontsize=8, loc='upper right', framealpha=0.3,
               labelcolor=WHITE, edgecolor=GRAY + '44')

    ax2.text(0.5, -0.12, '74% model size reduction  |  4.2x latency improvement',
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=7, color=GRAY, family='sans-serif')

    plt.tight_layout(pad=1.5)
    fig.savefig('assets/evaluation_results.png', dpi=120, facecolor=BG,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print('Saved: assets/evaluation_results.png')


if __name__ == '__main__':
    create_card_thumbnail()
    create_architecture_diagram()
    create_eval_image()
    print('\nAll images generated in assets/')
