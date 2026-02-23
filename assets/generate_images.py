"""
Generate professional writeup images for MedGemma Impact Challenge submission.

Clean, light-themed visuals suitable for competition media gallery.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Professional color palette ──
WHITE = '#FFFFFF'
BG_LIGHT = '#F8FAFC'       # very light gray
BG_CARD = '#FFFFFF'
TEXT_DARK = '#1E293B'       # slate-800
TEXT_MED = '#475569'        # slate-600
TEXT_LIGHT = '#94A3B8'      # slate-400
BORDER = '#E2E8F0'          # slate-200

# Agent colors (vibrant but professional)
BLUE = '#3B82F6'            # intake
INDIGO = '#6366F1'          # imaging
EMERALD = '#10B981'         # reasoning
ROSE = '#F43F5E'            # guidelines
AMBER = '#F59E0B'           # education

# Accent / section colors
TEAL = '#14B8A6'
SKY = '#0EA5E9'
VIOLET = '#8B5CF6'
ORANGE = '#F97316'

AGENT_COLORS = [BLUE, INDIGO, EMERALD, ROSE, AMBER]
AGENT_NAMES = ['Intake', 'Imaging', 'Reasoning', 'Guidelines', 'Education']
AGENT_SUBS = [
    'Structured HPI\n+ Red Flags',
    'MedGemma CXR\n+ MedSigLIP',
    'Differential Dx\n+ Workup',
    'RAG Evidence\nRetrieval',
    'Patient-Friendly\n3 Levels',
]


def _shadow_box(ax, x, y, w, h, color, alpha=0.08, offset=0.004):
    """Draw a subtle shadow behind a box."""
    shadow = FancyBboxPatch(
        (x + offset, y - offset), w, h,
        boxstyle="round,pad=0.008",
        facecolor='#000000',
        edgecolor='none',
        alpha=alpha,
        zorder=0,
    )
    ax.add_patch(shadow)


def _agent_box(ax, x, y, w, h, name, sublabel, color, fontsize=9, sublabel_size=6.5):
    """Draw a professional agent box with colored left border."""
    _shadow_box(ax, x, y, w, h, color)

    # White card
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008",
        facecolor=WHITE,
        edgecolor=BORDER,
        linewidth=0.8,
        zorder=1,
    )
    ax.add_patch(card)

    # Colored left accent bar
    bar_w = 0.006
    bar = FancyBboxPatch(
        (x + 0.003, y + 0.008), bar_w, h - 0.016,
        boxstyle="round,pad=0.002",
        facecolor=color,
        edgecolor='none',
        zorder=2,
    )
    ax.add_patch(bar)

    # Text
    ax.text(x + w/2 + 0.003, y + h * 0.65, name, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=TEXT_DARK, family='sans-serif', zorder=3)
    if sublabel:
        ax.text(x + w/2 + 0.003, y + h * 0.28, sublabel, ha='center', va='center',
                fontsize=sublabel_size, color=TEXT_LIGHT, family='sans-serif',
                linespacing=1.3, zorder=3)


def _arrow_right(ax, x1, y, x2, color=TEXT_LIGHT):
    """Draw a clean rightward arrow."""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                                connectionstyle='arc3,rad=0'),
                zorder=2)


def _section_label(ax, x, y, text, color, fontsize=8):
    """Draw a section label with colored dot."""
    ax.plot(x - 0.012, y, 'o', color=color, markersize=5, zorder=3)
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold',
            color=TEXT_DARK, family='sans-serif', va='center', zorder=3)


# ════════════════════════════════════════════════════════════════
# IMAGE 1: Card / Thumbnail (560 x 280)
# ════════════════════════════════════════════════════════════════
def create_card_thumbnail():
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.5), dpi=80)

    # Gradient background using imshow
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack([gradient] * 256)
    ax.imshow(gradient, aspect='auto', extent=[0, 1, 0, 1],
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                  'bg', ['#1E3A5F', '#0F172A']), zorder=0, alpha=0.95)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title with glow effect
    title = ax.text(0.5, 0.88, 'PrimaCare AI', ha='center', va='center',
                    fontsize=24, fontweight='bold', color=WHITE, family='sans-serif')
    title.set_path_effects([pe.withStroke(linewidth=3, foreground=BLUE + '44')])

    ax.text(0.5, 0.78, 'Multi-Agent CXR Diagnostic Support with Patient Education',
            ha='center', va='center', fontsize=9, color='#CBD5E1', family='sans-serif')

    # Pipeline boxes
    box_w = 0.125
    box_h = 0.2
    gap = 0.022
    total_w = 5 * box_w + 4 * gap
    sx = (1 - total_w) / 2
    y = 0.4

    for i, (name, color) in enumerate(zip(AGENT_NAMES, AGENT_COLORS)):
        x = sx + i * (box_w + gap)
        # Colored box with transparency
        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.008",
            facecolor=color + '33',
            edgecolor=color + 'AA',
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + box_w/2, y + box_h/2, name, ha='center', va='center',
                fontsize=8, fontweight='bold', color=WHITE, family='sans-serif')

        if i < 4:
            _arrow_right(ax, x + box_w + 0.003, y + box_h/2,
                         x + box_w + gap - 0.003, color='#64748B')

    # Model badges at bottom
    badges = [
        ('MedGemma 1.5 4B', BLUE),
        ('MedSigLIP 448', INDIGO),
        ('ONNX INT8 Edge', AMBER),
    ]
    badge_y = 0.15
    badge_w = 0.2
    badge_gap = 0.04
    total_bw = 3 * badge_w + 2 * badge_gap
    bsx = (1 - total_bw) / 2

    for i, (label, color) in enumerate(badges):
        bx = bsx + i * (badge_w + badge_gap)
        box = FancyBboxPatch(
            (bx, badge_y - 0.03), badge_w, 0.065,
            boxstyle="round,pad=0.006",
            facecolor=color + '22',
            edgecolor=color + '66',
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(bx + badge_w/2, badge_y + 0.003, label, ha='center', va='center',
                fontsize=7.5, color='#E2E8F0', family='sans-serif')

    # Track labels
    ax.text(0.5, 0.04, 'Main Track  |  Agentic Workflow  |  Novel Task  |  Edge AI',
            ha='center', va='center', fontsize=7, color='#64748B', family='sans-serif')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig('assets/card_thumbnail.png', dpi=80, facecolor='#0F172A',
                bbox_inches='tight', pad_inches=0)
    plt.close()
    print('Saved: assets/card_thumbnail.png')


# ════════════════════════════════════════════════════════════════
# IMAGE 2: Architecture Diagram (media gallery)
# ════════════════════════════════════════════════════════════════
def create_architecture_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.96, 'PrimaCare AI  —  System Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=TEXT_DARK, family='sans-serif')
    ax.text(0.5, 0.925, 'Multi-Agent Diagnostic Pipeline with Edge Deployment',
            ha='center', va='center', fontsize=9, color=TEXT_LIGHT, family='sans-serif')

    # ── CLOUD SECTION ──
    cloud_y = 0.60
    cloud_h = 0.26

    # Cloud background card
    _shadow_box(ax, 0.03, cloud_y - 0.02, 0.94, cloud_h, SKY, alpha=0.04, offset=0.005)
    cloud_bg = FancyBboxPatch(
        (0.03, cloud_y - 0.02), 0.94, cloud_h,
        boxstyle="round,pad=0.012",
        facecolor=SKY + '08',
        edgecolor=SKY + '33',
        linewidth=1.0,
        linestyle='--',
        zorder=0,
    )
    ax.add_patch(cloud_bg)
    _section_label(ax, 0.075, cloud_y + cloud_h - 0.04, 'CLOUD PIPELINE (GPU)', SKY, fontsize=8)

    # Agent boxes
    box_w = 0.145
    box_h = 0.15
    gap = 0.02
    total = 5 * box_w + 4 * gap
    sx = (1 - total) / 2
    agent_y = cloud_y + 0.01

    # Input label
    ax.text(sx - 0.04, agent_y + box_h/2, 'CXR\nImage\n+\nHistory', ha='center', va='center',
            fontsize=7, color=TEXT_LIGHT, family='sans-serif', linespacing=1.3)
    _arrow_right(ax, sx - 0.015, agent_y + box_h/2, sx - 0.003, TEXT_LIGHT)

    for i in range(5):
        x = sx + i * (box_w + gap)
        _agent_box(ax, x, agent_y, box_w, box_h, AGENT_NAMES[i], AGENT_SUBS[i],
                   AGENT_COLORS[i], fontsize=9, sublabel_size=6)
        if i < 4:
            _arrow_right(ax, x + box_w + 0.003, agent_y + box_h/2,
                         x + box_w + gap - 0.003, TEXT_LIGHT)

    # Output label
    last_x = sx + 4 * (box_w + gap) + box_w
    _arrow_right(ax, last_x + 0.003, agent_y + box_h/2, last_x + 0.02, TEXT_LIGHT)
    ax.text(last_x + 0.05, agent_y + box_h/2, 'Clinical\nReport +\nEducation', ha='center', va='center',
            fontsize=7, color=EMERALD, fontweight='bold', family='sans-serif', linespacing=1.3)

    # ── EDGE SECTION ──
    edge_y = 0.22
    edge_h = 0.26

    # Edge background card
    _shadow_box(ax, 0.03, edge_y - 0.02, 0.44, edge_h, AMBER, alpha=0.04, offset=0.005)
    edge_bg = FancyBboxPatch(
        (0.03, edge_y - 0.02), 0.44, edge_h,
        boxstyle="round,pad=0.012",
        facecolor=AMBER + '08',
        edgecolor=AMBER + '33',
        linewidth=1.0,
        linestyle='--',
        zorder=0,
    )
    ax.add_patch(edge_bg)
    _section_label(ax, 0.075, edge_y + edge_h - 0.04, 'EDGE SCREENING (CPU Only)', AMBER, fontsize=8)

    # Edge boxes
    ebox_w = 0.16
    ebox_h = 0.13
    _agent_box(ax, 0.07, edge_y + 0.01, ebox_w, ebox_h,
               'MedSigLIP', 'ONNX INT8\n422 MB', AMBER, fontsize=9, sublabel_size=6)
    _arrow_right(ax, 0.235, edge_y + 0.01 + ebox_h/2, 0.27, AMBER)
    _agent_box(ax, 0.275, edge_y + 0.01, ebox_w, ebox_h,
               'Binary Screen', 'Normal vs\nPneumonia', ORANGE, fontsize=9, sublabel_size=6)

    # Escalation arrow from edge to cloud
    ax.annotate('', xy=(0.5, cloud_y - 0.02), xytext=(0.395, edge_y + 0.14),
                arrowprops=dict(arrowstyle='->', color=EMERALD, lw=1.5,
                                connectionstyle='arc3,rad=-0.15'),
                zorder=3)
    ax.text(0.44, 0.50, 'Positive\nEscalate', fontsize=7, color=EMERALD,
            ha='center', va='center', fontweight='bold', family='sans-serif',
            linespacing=1.3)

    # ── KEY METRICS SECTION ──
    _shadow_box(ax, 0.52, edge_y - 0.02, 0.45, edge_h, TEAL, alpha=0.04, offset=0.005)
    metrics_bg = FancyBboxPatch(
        (0.52, edge_y - 0.02), 0.45, edge_h,
        boxstyle="round,pad=0.012",
        facecolor=TEAL + '08',
        edgecolor=TEAL + '33',
        linewidth=1.0,
        linestyle='--',
        zorder=0,
    )
    ax.add_patch(metrics_bg)
    _section_label(ax, 0.565, edge_y + edge_h - 0.04, 'KEY METRICS', TEAL, fontsize=8)

    metrics = [
        ('F1 Score:', '0.73', '(95% CI: 0.64 — 0.80)'),
        ('Recall:', '100%', '(zero missed pneumonia)'),
        ('Precision:', '57.5%', '(recall-priority threshold)'),
        ('Pipeline:', '~126s', '(GPU, full 5-agent)'),
        ('Edge:', '~4s', '(CPU only, binary)'),
    ]
    for j, (label, value, note) in enumerate(metrics):
        my = edge_y + edge_h - 0.09 - j * 0.038
        ax.text(0.57, my, label, fontsize=7.5, fontweight='bold',
                color=TEXT_MED, family='sans-serif', va='center')
        ax.text(0.65, my, value, fontsize=7.5, fontweight='bold',
                color=TEAL, family='sans-serif', va='center')
        ax.text(0.715, my, note, fontsize=6.5,
                color=TEXT_LIGHT, family='sans-serif', va='center')

    # ── MODEL STACK (bottom) ──
    model_y = 0.06
    models = [
        ('MedGemma 1.5 4B', 'Multimodal Analysis', BLUE),
        ('MedSigLIP 448', 'Zero-Shot CXR', INDIGO),
        ('all-MiniLM-L6-v2', 'Guidelines RAG', TEAL),
        ('ONNX Runtime', 'Edge Inference', AMBER),
    ]
    mw = 0.19
    mgap = 0.03
    total_mw = 4 * mw + 3 * mgap
    msx = (1 - total_mw) / 2

    for i, (name, role, color) in enumerate(models):
        mx = msx + i * (mw + mgap)
        box = FancyBboxPatch(
            (mx, model_y), mw, 0.075,
            boxstyle="round,pad=0.006",
            facecolor=color + '15',
            edgecolor=color + '55',
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(box)
        ax.text(mx + mw/2, model_y + 0.048, name, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=color, family='sans-serif', zorder=2)
        ax.text(mx + mw/2, model_y + 0.022, role, ha='center', va='center',
                fontsize=6.5, color=TEXT_LIGHT, family='sans-serif', zorder=2)

    plt.tight_layout(pad=0.5)
    fig.savefig('assets/architecture_diagram.png', dpi=150, facecolor=BG_LIGHT,
                bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print('Saved: assets/architecture_diagram.png')


# ════════════════════════════════════════════════════════════════
# IMAGE 3: Evaluation Results (media gallery)
# ════════════════════════════════════════════════════════════════
def create_eval_image():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=150,
                              gridspec_kw={'width_ratios': [1, 1, 0.8]})
    fig.patch.set_facecolor(BG_LIGHT)
    fig.suptitle('PrimaCare AI  —  Evaluation Results',
                 fontsize=14, fontweight='bold', color=TEXT_DARK,
                 family='sans-serif', y=0.98)

    # ── Panel 1: Classification Metrics ──
    ax = axes[0]
    ax.set_facecolor(WHITE)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    ax.set_title('Binary Pneumonia Classification', fontsize=10,
                  fontweight='bold', color=TEXT_DARK, pad=12, family='sans-serif')

    categories = ['F1 Score', 'Recall', 'Precision', 'Accuracy', 'Specificity']
    values = [0.73, 1.0, 0.575, 0.63, 0.26]
    bar_colors = [EMERALD, BLUE, INDIGO, SKY, VIOLET]

    bars = ax.barh(categories, values, color=bar_colors, height=0.55, zorder=2)

    # Light gridlines
    ax.set_xlim(0, 1.18)
    ax.xaxis.grid(True, alpha=0.15, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MED, labelsize=8)
    ax.tick_params(axis='y', length=0)

    for bar, val, color in zip(bars, values, bar_colors):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8.5, fontweight='bold',
                color=color, family='sans-serif')

    ax.text(0.5, -0.1, '100 balanced samples  |  Threshold: 0.30 (recall priority)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=7, color=TEXT_LIGHT, family='sans-serif')
    ax.text(0.5, -0.17, 'F1 95% CI: [0.64, 0.80]',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=7, color=EMERALD, fontweight='bold', family='sans-serif')

    # ── Panel 2: Pipeline Latency ──
    ax2 = axes[1]
    ax2.set_facecolor(WHITE)
    for spine in ax2.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    ax2.set_title('Pipeline Latency (Kaggle T4)', fontsize=10,
                   fontweight='bold', color=TEXT_DARK, pad=12, family='sans-serif')

    stages = ['Intake', 'Imaging', 'Reasoning', 'Guidelines', 'Education']
    latencies = [23.3, 16.9, 38.0, 32.9, 15.0]
    stage_colors = AGENT_COLORS

    bars2 = ax2.barh(stages, latencies, color=stage_colors, height=0.55, zorder=2)

    ax2.set_xlim(0, 52)
    ax2.set_xlabel('Seconds', fontsize=8, color=TEXT_MED)
    ax2.xaxis.grid(True, alpha=0.15, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(colors=TEXT_MED, labelsize=8)
    ax2.tick_params(axis='y', length=0)

    for bar, val, color in zip(bars2, latencies, stage_colors):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}s', va='center', fontsize=8, fontweight='bold',
                 color=color, family='sans-serif')

    total_time = sum(latencies)
    ax2.text(0.5, -0.1, f'Total pipeline: ~{total_time:.0f}s end-to-end',
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=7, color=TEXT_LIGHT, family='sans-serif')

    # ── Panel 3: Edge vs Cloud Comparison ──
    ax3 = axes[2]
    ax3.set_facecolor(WHITE)
    for spine in ax3.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    ax3.set_title('Edge vs Cloud', fontsize=10,
                   fontweight='bold', color=TEXT_DARK, pad=12, family='sans-serif')

    # Comparison table style
    comparisons = [
        ('Compute', 'GPU (T4)', 'CPU only'),
        ('Model Size', '~3.5 GB', '422 MB'),
        ('Latency', '~17s', '~4s'),
        ('Recall', '100%', 'TBD'),
        ('Format', 'PyTorch', 'ONNX INT8'),
    ]

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # Header row
    header_y = 0.88
    ax3.text(0.15, header_y, '', fontsize=8, fontweight='bold',
             color=TEXT_DARK, family='sans-serif', va='center')
    ax3.text(0.55, header_y, 'Cloud', fontsize=8, fontweight='bold',
             color=SKY, family='sans-serif', va='center', ha='center')
    ax3.text(0.85, header_y, 'Edge', fontsize=8, fontweight='bold',
             color=AMBER, family='sans-serif', va='center', ha='center')

    # Separator
    ax3.plot([0.05, 0.95], [header_y - 0.04, header_y - 0.04],
             color=BORDER, linewidth=0.8)

    for j, (label, cloud_val, edge_val) in enumerate(comparisons):
        row_y = header_y - 0.08 - j * 0.14
        # Alternate row background
        if j % 2 == 0:
            row_bg = FancyBboxPatch(
                (0.03, row_y - 0.05), 0.94, 0.1,
                boxstyle="round,pad=0.005",
                facecolor=BG_LIGHT,
                edgecolor='none',
            )
            ax3.add_patch(row_bg)

        ax3.text(0.08, row_y, label, fontsize=7.5, fontweight='bold',
                 color=TEXT_MED, family='sans-serif', va='center')
        ax3.text(0.55, row_y, cloud_val, fontsize=7.5,
                 color=TEXT_DARK, family='sans-serif', va='center', ha='center')
        ax3.text(0.85, row_y, edge_val, fontsize=7.5,
                 color=TEXT_DARK, family='sans-serif', va='center', ha='center')

    # Size reduction callout
    ax3.text(0.5, 0.06, '74% size reduction', fontsize=8, fontweight='bold',
             color=EMERALD, family='sans-serif', va='center', ha='center')

    plt.tight_layout(pad=1.5)
    fig.savefig('assets/evaluation_results.png', dpi=150, facecolor=BG_LIGHT,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print('Saved: assets/evaluation_results.png')


# ════════════════════════════════════════════════════════════════
# IMAGE 4: Patient Education Showcase (media gallery)
# ════════════════════════════════════════════════════════════════
def create_education_image():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Novel Task  —  Patient Education System',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=TEXT_DARK, family='sans-serif')
    ax.text(0.5, 0.905, 'Translating clinical reports into patient-friendly language at 3 reading levels',
            ha='center', va='center', fontsize=9, color=TEXT_LIGHT, family='sans-serif')

    # Three cards for three reading levels
    levels = [
        {
            'title': 'Basic',
            'audience': '6th-grade reading level',
            'color': EMERALD,
            'example': '"You have a lung infection\nthat is making it hard\nto breathe. You need\nmedicine to help fight\nthe infection."',
        },
        {
            'title': 'Intermediate',
            'audience': 'General adult',
            'color': BLUE,
            'example': '"You have community-acquired\npneumonia — a lung infection\ncausing consolidation in\nyour right lower lobe.\nAntibiotics are recommended."',
        },
        {
            'title': 'Detailed',
            'audience': 'Patients wanting depth',
            'color': VIOLET,
            'example': '"Right lower lobe consolidation\nconsistent with community-\nacquired pneumonia (CAP).\nConsolidation = lung tissue\nfilled with fluid/pus."',
        },
    ]

    card_w = 0.27
    card_h = 0.52
    card_gap = 0.04
    total_cw = 3 * card_w + 2 * card_gap
    csx = (1 - total_cw) / 2
    card_y = 0.22

    for i, level in enumerate(levels):
        cx = csx + i * (card_w + card_gap)
        color = level['color']

        # Card shadow
        _shadow_box(ax, cx, card_y, card_w, card_h, color, alpha=0.06, offset=0.005)

        # Card
        card = FancyBboxPatch(
            (cx, card_y), card_w, card_h,
            boxstyle="round,pad=0.012",
            facecolor=WHITE,
            edgecolor=BORDER,
            linewidth=0.8,
            zorder=1,
        )
        ax.add_patch(card)

        # Colored header bar
        header = FancyBboxPatch(
            (cx + 0.005, card_y + card_h - 0.085), card_w - 0.01, 0.075,
            boxstyle="round,pad=0.008",
            facecolor=color + '18',
            edgecolor='none',
            zorder=2,
        )
        ax.add_patch(header)

        # Level title
        ax.text(cx + card_w/2, card_y + card_h - 0.035, level['title'],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color=color, family='sans-serif', zorder=3)
        ax.text(cx + card_w/2, card_y + card_h - 0.065, level['audience'],
                ha='center', va='center', fontsize=7, color=TEXT_LIGHT,
                family='sans-serif', zorder=3)

        # Example text box
        example_bg = FancyBboxPatch(
            (cx + 0.015, card_y + 0.1), card_w - 0.03, 0.32,
            boxstyle="round,pad=0.008",
            facecolor=BG_LIGHT,
            edgecolor=BORDER,
            linewidth=0.5,
            zorder=2,
        )
        ax.add_patch(example_bg)

        ax.text(cx + card_w/2, card_y + 0.26, level['example'],
                ha='center', va='center', fontsize=7, color=TEXT_MED,
                family='sans-serif', style='italic', linespacing=1.4, zorder=3)

        # Label
        ax.text(cx + card_w/2, card_y + 0.05, 'Example Output',
                ha='center', va='center', fontsize=6.5, color=TEXT_LIGHT,
                family='sans-serif', zorder=3)

    # Bottom: output structure
    structure_items = [
        'Simplified Diagnosis',
        'What It Means',
        'Next Steps',
        'When to Seek Help',
        'Glossary',
    ]
    item_colors = [BLUE, INDIGO, EMERALD, ROSE, AMBER]

    ax.text(0.5, 0.14, 'Structured Output:', ha='center', va='center',
            fontsize=8, fontweight='bold', color=TEXT_DARK, family='sans-serif')

    total_items_w = 0.85
    item_w = total_items_w / len(structure_items)
    isx = (1 - total_items_w) / 2

    for i, (item, color) in enumerate(zip(structure_items, item_colors)):
        ix = isx + i * item_w
        ax.plot(ix + item_w/2 - 0.03, 0.06, 's', color=color, markersize=6, zorder=3)
        ax.text(ix + item_w/2, 0.06, item, ha='center', va='center',
                fontsize=6.5, color=TEXT_MED, family='sans-serif')

    plt.tight_layout(pad=0.5)
    fig.savefig('assets/education_showcase.png', dpi=150, facecolor=BG_LIGHT,
                bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print('Saved: assets/education_showcase.png')


# ════════════════════════════════════════════════════════════════
# IMAGE 5: Tiered Deployment Flow (media gallery)
# ════════════════════════════════════════════════════════════════
def create_deployment_flow():
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor(BG_LIGHT)
    ax.set_facecolor(BG_LIGHT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title
    ax.text(0.5, 0.94, 'Edge AI  —  Tiered Deployment Architecture',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=TEXT_DARK, family='sans-serif')
    ax.text(0.5, 0.89, 'CPU-only screening with intelligent cloud escalation',
            ha='center', va='center', fontsize=9, color=TEXT_LIGHT, family='sans-serif')

    # Flow: CXR Input -> Edge Classifier -> Decision -> Cloud Pipeline / Done
    # Step boxes
    box_h = 0.2
    step_y = 0.48

    # Step 1: Input
    _agent_box(ax, 0.02, step_y, 0.13, box_h, 'CXR Image', 'Patient\nChest X-Ray', TEXT_MED,
               fontsize=9, sublabel_size=6.5)

    _arrow_right(ax, 0.155, step_y + box_h/2, 0.18, TEXT_LIGHT)

    # Step 2: Edge Classifier
    _agent_box(ax, 0.185, step_y, 0.18, box_h, 'Edge Classifier', 'MedSigLIP ONNX\nINT8 (422 MB)', AMBER,
               fontsize=9, sublabel_size=6.5)

    _arrow_right(ax, 0.37, step_y + box_h/2, 0.40, TEXT_LIGHT)

    # Step 3: Decision diamond (simulated with rotated box)
    dx, dy = 0.435, step_y + box_h/2
    diamond_size = 0.035
    diamond = plt.Polygon([
        (dx, dy + diamond_size),
        (dx + diamond_size, dy),
        (dx, dy - diamond_size),
        (dx - diamond_size, dy),
    ], facecolor=AMBER + '22', edgecolor=AMBER, linewidth=1.2, zorder=2)
    ax.add_patch(diamond)
    ax.text(dx, dy, '?', ha='center', va='center', fontsize=10, fontweight='bold',
            color=AMBER, family='sans-serif', zorder=3)

    # Branch: Positive -> Cloud
    ax.annotate('', xy=(0.55, step_y + box_h/2 + 0.06),
                xytext=(dx + diamond_size + 0.005, dy + 0.01),
                arrowprops=dict(arrowstyle='->', color=ROSE, lw=1.5,
                                connectionstyle='arc3,rad=-0.1'),
                zorder=2)
    ax.text(0.505, step_y + box_h/2 + 0.1, 'Pneumonia\nDetected', fontsize=7,
            color=ROSE, fontweight='bold', ha='center', va='center',
            family='sans-serif', linespacing=1.3)

    # Cloud pipeline box
    _agent_box(ax, 0.555, step_y + 0.05, 0.25, box_h - 0.05, 'Full 5-Agent Pipeline',
               'MedGemma Cloud\nDiagnosis + Education', SKY, fontsize=9, sublabel_size=6.5)

    _arrow_right(ax, 0.81, step_y + box_h/2 + 0.025, 0.84, TEXT_LIGHT)

    # Cloud output
    _agent_box(ax, 0.845, step_y + 0.05, 0.14, box_h - 0.05, 'Full Report',
               'Clinical +\nEducation', EMERALD, fontsize=9, sublabel_size=6.5)

    # Branch: Normal -> Done
    ax.annotate('', xy=(0.555, step_y - 0.08),
                xytext=(dx + diamond_size + 0.005, dy - 0.01),
                arrowprops=dict(arrowstyle='->', color=EMERALD, lw=1.5,
                                connectionstyle='arc3,rad=0.1'),
                zorder=2)
    ax.text(0.505, step_y - 0.07, 'Normal\n(High Conf)', fontsize=7,
            color=EMERALD, fontweight='bold', ha='center', va='center',
            family='sans-serif', linespacing=1.3)

    # Normal result box
    _agent_box(ax, 0.56, step_y - 0.17, 0.18, 0.12, 'Normal Result',
               'No escalation needed', EMERALD, fontsize=9, sublabel_size=6.5)

    # Stats bar at bottom
    stats = [
        ('Model', 'MedSigLIP 448 (ONNX INT8)'),
        ('Size', '422 MB (74% reduction)'),
        ('Latency', '~4s per image (CPU)'),
        ('Zero GPU', 'Required for screening'),
    ]

    stat_w = 0.2
    stat_gap = 0.035
    total_sw = 4 * stat_w + 3 * stat_gap
    ssx = (1 - total_sw) / 2
    stat_y = 0.08

    for i, (label, value) in enumerate(stats):
        sx = ssx + i * (stat_w + stat_gap)
        box = FancyBboxPatch(
            (sx, stat_y), stat_w, 0.08,
            boxstyle="round,pad=0.006",
            facecolor=AMBER + '10',
            edgecolor=AMBER + '33',
            linewidth=0.5,
            zorder=1,
        )
        ax.add_patch(box)
        ax.text(sx + stat_w/2, stat_y + 0.052, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color=AMBER, family='sans-serif', zorder=2)
        ax.text(sx + stat_w/2, stat_y + 0.025, value, ha='center', va='center',
                fontsize=6.5, color=TEXT_MED, family='sans-serif', zorder=2)

    plt.tight_layout(pad=0.5)
    fig.savefig('assets/edge_deployment.png', dpi=150, facecolor=BG_LIGHT,
                bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print('Saved: assets/edge_deployment.png')


if __name__ == '__main__':
    create_card_thumbnail()
    create_architecture_diagram()
    create_eval_image()
    create_education_image()
    create_deployment_flow()
    print('\nAll images generated in assets/')
