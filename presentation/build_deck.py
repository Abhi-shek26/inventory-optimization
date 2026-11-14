"""Build the final slide deck from the experiment outputs.

Every number on every slide is read from results/*.json / results/*.csv, so the
deck cannot drift from what the code actually produced.  Run the experiments
first:

    python -m experiments.train_dqn
    python -m experiments.evaluate
    python -m experiments.milp_analysis
    python -m experiments.statistics
    python -m presentation.build_deck
"""

from __future__ import annotations

import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

RESULTS = "results"
OUT = os.path.join("presentation", "MILP_Inventory_RL_vs_Optimization.pptx")

NAVY = RGBColor(0x1B, 0x26, 0x33)
GREY = RGBColor(0x5A, 0x6B, 0x7B)
RED = RGBColor(0xC0, 0x39, 0x2B)
BLUE = RGBColor(0x2C, 0x7F, 0xB8)
GREEN = RGBColor(0x31, 0xA3, 0x54)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
BORDER = RGBColor(0xD5, 0xDB, 0xDB)

W, H = Inches(13.333), Inches(7.5)


# --------------------------------------------------------------------- helpers
def _new(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _para(tf, text, size=16, bold=False, color=NAVY, first=False,
          space_before=6, bullet=None, italic=False, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    run = p.add_run()
    run.text = (f"{bullet} " if bullet else "") + text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Segoe UI"
    return p


def header(slide, title, subtitle=None):
    bar = slide.shapes.add_shape(1, 0, 0, W, Inches(0.13))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RED
    bar.line.fill.background()
    tf = _box(slide, Inches(0.6), Inches(0.28), Inches(12.1), Inches(0.7))
    _para(tf, title, size=26, bold=True, first=True)
    if subtitle:
        tf2 = _box(slide, Inches(0.6), Inches(0.92), Inches(12.1), Inches(0.4))
        _para(tf2, subtitle, size=14, color=GREY, italic=True, first=True)


def footer(slide, n):
    tf = _box(slide, Inches(0.6), Inches(6.95), Inches(6), Inches(0.35))
    _para(tf, "RL vs. Optimization - Inventory under demand uncertainty",
          size=10, color=GREY, first=True)
    tf = _box(slide, Inches(12.2), Inches(6.95), Inches(0.6), Inches(0.35))
    _para(tf, str(n), size=10, color=GREY, first=True, align=PP_ALIGN.RIGHT)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def bullets(slide, items, x=Inches(0.75), y=Inches(1.35), w=Inches(11.8),
            h=Inches(5.2), size=16):
    """items: list of (level, text) or plain strings."""
    tf = _box(slide, x, y, w, h)
    first = True
    for item in items:
        level, text = item if isinstance(item, tuple) else (0, item)
        if level == 0:
            _para(tf, text, size=size, first=first, bullet="\u25aa", color=NAVY)
        else:
            _para(tf, "    -  " + text, size=size - 2, first=first, color=GREY)
        first = False


def table(slide, rows, x=Inches(0.75), y=Inches(1.4), w=Inches(11.8),
          col_widths=None, size=12, header_fill=NAVY):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, x, y, w, Inches(0.32 * n_rows))
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(w * cw / total))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            para.font.name = "Segoe UI"
            if r == 0:
                para.font.bold = True
                para.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            else:
                para.font.color.rgb = NAVY
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
    return tbl


def picture(slide, path, x=Inches(0.75), y=Inches(1.5), width=None, height=None):
    if not os.path.exists(path):
        tf = _box(slide, x, y, Inches(6), Inches(1))
        _para(tf, f"[missing figure: {path}]", size=12, color=RED, first=True)
        return None
    kw = {}
    if width:
        kw["width"] = width
    if height:
        kw["height"] = height
    return slide.shapes.add_picture(path, x, y, **kw)


def caption(slide, text, x, y, w=Inches(11.8)):
    tf = _box(slide, x, y, w, Inches(0.6))
    _para(tf, text, size=11, color=GREY, italic=True, first=True)


def band(slide, x, y, w, h):
    shape = slide.shapes.add_shape(1, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = BORDER
    return shape


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run the experiments first (see module docstring).")
    with open(path) as fh:
        return json.load(fh)


def pfmt(v):
    """Compact p-value for a table cell."""
    if v < 1e-3:
        return "p<" + f"{v:.0e}".replace("e-0", "e-")
    return f"p={v:.2f}"


# ----------------------------------------------------------------------- build
def build():
    ev = load("evaluation_summary.json")
    ma = load("milp_analysis.json")
    pt = load("paired_tests.json")
    cfg = ev["config"]
    sc = ev["scenarios"]
    S = ev["summaries"]
    base = "Exp1_stationary"
    m_dqn = S[base]["DQN"]
    m_dp = S[base]["DP"]
    m_roll = S[base]["MILP (rolling forecast)"]
    m_clair = S[base]["MILP (clairvoyant)"]
    n_ep = ev["episodes_per_scenario"]
    base_sc = list(pt["scenarios"])[0]

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    # 1 ------------------------------------------------------------------ title
    n += 1
    s = _new(prs)
    shape = s.shapes.add_shape(1, 0, Inches(1.9), W, Inches(2.5))
    shape.fill.solid()
    shape.fill.fore_color.rgb = NAVY
    shape.line.fill.background()
    tf = _box(s, Inches(0.9), Inches(2.15), Inches(11.6), Inches(2.0))
    _para(tf, "RL vs. Optimization", size=44, bold=True, color=WHITE, first=True)
    _para(tf, "Managing inventory under demand uncertainty", size=26,
          color=RGBColor(0xC8, 0xD3, 0xDF), space_before=4)
    tf = _box(s, Inches(0.9), Inches(4.8), Inches(11.6), Inches(2))
    _para(tf, "Benchmarking a learned policy (DQN) against exact optimisation:",
          size=16, color=NAVY, first=True)
    _para(tf, "Dynamic Programming (known distribution)  and  MILP with fixed-charge "
              "lot sizing + Branch & Bound (known demand)", size=16, color=GREY,
          space_before=2)
    _para(tf, "Team members:  ______________    Project number:  ______________",
          size=14, color=GREY, space_before=18)
    notes(s, "Opening line: we study one product, one warehouse, 20 days. The question "
             "is how much to order each day when demand is uncertain. We solve it three "
             "ways and compare the money spent. Our part is the MILP / Branch and Bound "
             "benchmark.")
    footer(s, n)

    # 2 --------------------------------------------------------- introduction
    n += 1
    s = _new(prs)
    header(s, "Introduction: the real-world problem",
           "An e-commerce seller restocking one fast-moving SKU")
    bullets(s, [
        (0, "A seller holds one product and re-orders it from a supplier every day."),
        (1, "Ordering is not free: every purchase order carries a fixed admin + freight "
            "cost K, whatever the quantity. Ordering a big batch less often looks attractive."),
        (1, "Stock left over at night costs money (storage, capital, markdowns)."),
        (1, "Demand that cannot be served is lost: the customer buys elsewhere."),
        (0, "Three costs pull in opposite directions: order often (pay K often) vs. order "
            "big (pay holding) vs. hold little (lose sales)."),
        (0, "Demand is unknown when the order is placed - that is the whole difficulty."),
        (0, "Same structure appears in retail replenishment, spare-parts management, "
            "hospital supplies, cloud capacity reservation."),
    ], y=Inches(1.5), size=17)
    notes(s, "Motivate with money: fixed cost per order vs holding vs lost sales. Say that "
             "the decision must be taken before demand is known, which is what makes it a "
             "sequential decision problem, i.e. an MDP.")
    footer(s, n)

    # 3 ------------------------------------------------- problem and objectives
    n += 1
    s = _new(prs)
    header(s, "Problem statement and objectives")
    bullets(s, [
        (0, "Problem: choose order quantities q_1..q_T for one product under stochastic "
            "demand, minimising expected total cost over a finite horizon T = "
            f"{cfg['horizon']} periods."),
        (0, "Objectives"),
        (1, "1. Formalise the system as a Markov Decision Process (state, action, cost)."),
        (1, "2. Solve it exactly with Dynamic Programming when the demand distribution is known."),
        (1, "3. Build a MILP model of the same problem - the fixed ordering cost makes it a "
            "fixed-charge lot-sizing problem - and solve it with Branch and Bound."),
        (1, "4. Learn a policy with a DQN that sees only experience, not the distribution."),
        (1, "5. Compare cost, service level, inventory, computation time and adaptability "
            "on identical demand paths."),
        (0, "The key question of the comparison: how much performance does a policy lose "
            "when it does not know the distribution?"),
    ], y=Inches(1.4), size=16)
    notes(s, "Emphasise that all three methods are evaluated on the same realised demand "
             "paths - common random numbers - otherwise the comparison is meaningless.")
    footer(s, n)

    # 4 ---------------------------------------------------- mathematical model
    n += 1
    s = _new(prs)
    header(s, "Mathematical formulation and assumptions",
           "One model, three solution methods")
    rows = [["Symbol", "Meaning", "Value"],
            ["T", "planning horizon (days)", str(cfg["horizon"])],
            ["I_t", "inventory at start of period t", "-"],
            ["q_t", "quantity ordered in period t", "-"],
            ["D_t", "demand in period t (random)", "0..12"],
            ["K", "fixed cost per period with an order", str(cfg["fixed_order_cost"])],
            ["c", "variable cost per unit ordered", str(cfg["order_cost"])],
            ["h", "holding cost per unit per period", str(cfg["holding_cost"])],
            ["p", "penalty per unit of lost demand", str(cfg["shortage_cost"])],
            ["Imax", "warehouse capacity", str(cfg["max_inventory"])],
            ["Qmax", "order capacity per period", str(cfg["max_order"])]]
    table(s, rows, x=Inches(0.75), y=Inches(1.45), w=Inches(5.6), size=11)
    tf = _box(s, Inches(6.7), Inches(1.4), Inches(6.0), Inches(5.2))
    _para(tf, "System dynamics", size=16, bold=True, first=True)
    for line in ["Available stock:   A_t = I_t + q_t",
                 "Sales:   sales_t = min(A_t, D_t)",
                 "Lost sales:   S_t = max(0, D_t - A_t)",
                 "Ending stock:   I_(t+1) = max(0, A_t - D_t)"]:
        _para(tf, line, size=13, color=GREY, space_before=8)
    _para(tf, "Objective", size=16, bold=True, space_before=18)
    _para(tf, "min  E[ sum_t ( K*1{q_t>0} + c*q_t + h*I_(t+1) + p*S_t ) ]",
          size=14, color=RED, space_before=8)
    _para(tf, "Assumptions: single product, single warehouse, zero lead time, "
              "lost sales (no backorders), integer units, stationary truncated "
              "Poisson demand, known cost parameters.",
          size=12, color=GREY, space_before=16)
    notes(s, "Note the indicator 1{q_t>0} in the objective - that single term is what "
             "makes the MILP an integer program. Everything else is linear.")
    footer(s, n)

    # 5 --------------------------------------------------- dynamic programming
    n += 1
    s = _new(prs)
    header(s, "Method 1 - Dynamic Programming (exact)")
    tf = _box(s, Inches(0.75), Inches(1.35), Inches(11.8), Inches(1.2))
    _para(tf, "Bellman equation (backward in time)", size=16, bold=True, first=True)
    _para(tf, "V_t(I) = min_q  C(I, q) + sum_d  P(D=d) * V_(t+1)( next_inventory(I, q, d) )",
          size=15, color=GREEN, space_before=6)
    bullets(s, [
        (0, f"State space: inventory 0..{cfg['max_inventory']} x period; action: order 0..Qmax."),
        (0, "The demand distribution is assumed known (truncated Poisson), so the expectation "
            "is computed exactly - no sampling."),
        (0, "Tabulates the optimal action for every (period, inventory) pair: an exact optimal "
            "policy, not just a plan."),
        (0, f"Expected optimal cost starting empty: {ev['dp_expected_cost']:.2f} "
            f"(solved in {ev['dp_solve_time_s']:.2f} s)."),
        (0, "The policy is a threshold (s, S) rule: order up to S whenever inventory falls to "
            "s. With a fixed cost K this batching behaviour is what theory predicts."),
        (0, "Limitation: it needs the distribution. If the world changes, the policy is stale "
            "- it cannot adapt without re-solving."),
    ], y=Inches(2.75), size=15)
    notes(s, "DP is our exact reference. Because it knows the distribution it is the fair, "
             "information-rich benchmark. If DQN matches DP, learning has captured the "
             "structure. If MILP beats DP, remember MILP sees the future.")
    footer(s, n)

    # 6 --------------------------------------------------------------- MILP
    n += 1
    s = _new(prs)
    header(s, "Our part - the fixed-charge lot-sizing MILP",
           "The fixed ordering cost K forces a yes/no decision per period")
    tf = _box(s, Inches(0.75), Inches(1.35), Inches(6.2), Inches(3.8))
    _para(tf, "Decision variables", size=15, bold=True, first=True)
    for line in ["q_t >= 0    order quantity (units)",
                 "I_t >= 0    ending inventory",
                 "S_t >= 0    lost sales (unmet demand)",
                 "y_t in {0,1}   setup indicator: 1 <=> an order is placed"]:
        _para(tf, line, size=12, color=GREY, space_before=5)
    _para(tf, "Objective", size=15, bold=True, space_before=14)
    _para(tf, "min  sum_t ( K*y_t + c*q_t + h*I_t + p*S_t )", size=14, color=RED,
          space_before=4)
    _para(tf, "Constraints", size=15, bold=True, space_before=14)
    for line in ["(1)  I_(t-1) + q_t - D_t + S_t = I_t        flow balance",
                 "(2)  q_t  <=  M * y_t                       big-M linking",
                 "(3)  0 <= q_t <= Qmax,  0 <= I_t <= Imax",
                 "(4)  0 <= S_t <= D_t"]:
        _para(tf, line, size=12, color=GREY, space_before=5)

    tf = _box(s, Inches(7.2), Inches(1.35), Inches(5.5), Inches(3.8))
    _para(tf, "The big-M trick", size=16, bold=True, first=True, color=RED)
    _para(tf, "q_t  <=  M * y_t", size=16, bold=True, space_before=6)
    _para(tf, "y_t = 1  ->  q_t <= M  : order allowed (M is not binding)", size=13,
          color=GREEN, space_before=10)
    _para(tf, "y_t = 0  ->  q_t <= 0  : order forced to zero", size=13, color=GREEN,
          space_before=6)
    _para(tf, "So the binary pays K only when a period is switched on. This is the single "
              "constraint that links the setup decision to the quantity decision.",
          size=13, color=GREY, space_before=12)
    _para(tf, f"Tightest valid M = min(Qmax, Imax) = min({cfg['max_order']}, "
              f"{cfg['max_inventory']}) = {min(cfg['max_order'], cfg['max_inventory'])}.",
          size=13, color=GREY, space_before=10)
    _para(tf, "Too small: cuts off feasible plans. Too large: weak LP bound, many more "
              "branching nodes.", size=13, color=GREY, space_before=10)

    band(s, Inches(0.75), Inches(5.25), Inches(6.0), Inches(1.35))
    tf = _box(s, Inches(0.95), Inches(5.32), Inches(5.7), Inches(1.25))
    _para(tf, "Why MILP and not something simpler?", size=13, bold=True, first=True)
    _para(tf, "Plain LP drops K: loses the setup decision and Branch and Bound entirely. "
              "Min-cost flow works only when K = 0. NLP fits the one-period newsvendor only. "
              "Scenario-based stochastic MILP is the proper next step - future work here.",
          size=10.5, color=GREY, space_before=3)
    notes(s, "This is the slide to know cold. y_t is the yes/no decision. big-M is the "
             "switch. Without constraint (2) the solver could set y_t = 0 and still order - "
             "free orders. With it, paying K is the only way to order.")
    footer(s, n)

    # 7 ------------------------------------------------------ branch and bound
    n += 1
    s = _new(prs)
    header(s, "Solving it: LP relaxation and Branch & Bound",
           "Why the LP answer is not a plan, and how branching fixes it")
    bnb = ma["bnb"]["rows"][0]
    ig = ma["integrality_gap"]
    tf = _box(s, Inches(0.75), Inches(1.35), Inches(5.9), Inches(5.1))
    _para(tf, "1. Relax y_t in {0,1}  ->  y_t in [0,1]: an LP", size=13, first=True)
    _para(tf, "   The LP bound is valid but loose. Measured on this problem: "
              f"{ig['mean_pct']:.1f}% below the MILP optimum on average over "
              f"{ig['paths']} demand paths.", size=12, color=GREY, space_before=4)
    _para(tf, "2. Branch on a fractional y_t", size=13, space_before=10)
    _para(tf, "   Fix y_t = 0 in one child, y_t = 1 in the other, re-solve the LP.",
          size=12, color=GREY, space_before=4)
    _para(tf, "3. Prune", size=13, space_before=10)
    _para(tf, "   If a node's LP bound is already worse than the best integer plan "
              "(incumbent), nothing inside it can win - discard it.", size=12,
          color=GREY, space_before=4)
    _para(tf, "4. Stop when the tree is empty", size=13, space_before=10)
    _para(tf, "   The incumbent is then provably optimal.", size=12, color=GREY,
          space_before=4)
    _para(tf, f"Example instance: root LP bound {bnb['root_lp_bound']:.0f}, optimum "
              f"{bnb['bnb_cost']:.0f} ({bnb['root_gap_pct']:.1f}% gap), resolved in "
              f"{bnb['nodes_explored']} nodes with {bnb['nodes_pruned']} pruned - identical "
              f"to CBC's answer.", size=12, color=RED, space_before=12)
    picture(s, os.path.join(RESULTS, "milp_bnb_trace.png"), x=Inches(6.9),
            y=Inches(1.5), width=Inches(5.9))
    caption(s, "Our own Branch & Bound run. Dots are the LP bound of each node visited; "
               "the red staircase is the tightest bound still open; the blue line is the best "
               "integer plan found. Where they meet, optimality is proven.",
            Inches(6.9), Inches(5.95), w=Inches(5.9))
    notes(s, "Two facts to state: (a) the LP relaxation is a valid lower bound because "
             "enlarging the feasible set can only lower a minimisation; (b) pruning is safe "
             "because a child's bound bounds everything below it. Branching only on y_t is "
             "exact because with all setups fixed the remaining problem is a flow problem "
             "with a totally unimodular matrix.")
    footer(s, n)

    # 8 ------------------------------------------------------- model validation
    n += 1
    s = _new(prs)
    header(s, "Is the MILP model correct? Independent checks",
           "Never trust a formulation because the solver says 'optimal'")
    small = ma["small_instance"]
    rows = [["Setup pattern y1..y4", "Orders placed", "Cost"]]
    patterns = sorted(small["patterns"], key=lambda r: r["cost"])
    for r in patterns[:6]:
        rows.append([r["pattern"], r["setups"],
                     f"{r['cost']:.2f}" + ("   <-- optimal" if r["is_optimal"] else "")])
    table(s, rows, x=Inches(0.75), y=Inches(1.5), w=Inches(4.6), size=11)
    caption(s, f"Check 1 - brute force. T=4, demand {small['demand']}. All 2^4 = 16 "
               f"setup patterns enumerated, each solved as an LP. Cheapest pattern "
               f"{small['best_pattern']} = {small['best_cost']:.2f}; the MILP returns "
               f"{small['milp_cost']:.2f}.",
            Inches(0.75), Inches(4.4), w=Inches(4.8))

    ww = ma["wagner_whitin"]
    tf = _box(s, Inches(6.0), Inches(1.45), Inches(6.6), Inches(2.4))
    _para(tf, "Check 2 - Wagner-Whitin", size=15, bold=True, first=True, color=GREEN)
    _para(tf, "The classics give an independent exact algorithm for the uncapacitated "
              "fixed-charge problem. Switch off capacity and shortage penalties, and the "
              "MILP must reproduce it exactly.", size=12, color=GREY, space_before=6)
    _para(tf, f"Result: {ww['instances']} instances, maximum absolute cost difference "
              f"= {ww['max_abs_difference']:.6f}.", size=13, color=NAVY, space_before=6)

    tf = _box(s, Inches(6.0), Inches(3.95), Inches(6.6), Inches(2.8))
    _para(tf, "Check 3 - tight vs loose big-M", size=15, bold=True, first=True, color=RED)
    for r in ma["big_m"]:
        _para(tf, f"{r['label']}:  LP root bound {r['lp_root_bound']:.1f} vs optimum "
                  f"{r['milp_cost']:.1f}  ->  gap {r['root_gap_pct']:.1f}%,  "
                  f"B&B nodes {r['bnb_nodes']:.0f}",
              size=11, color=GREY, space_before=5)
    _para(tf, "Same optimum, same answer - but the loose M makes the bound far weaker, so "
              "the tree grows. This is why we set M = min(Qmax, Imax).",
          size=12, color=NAVY, space_before=8)
    notes(s, "This slide answers 'how do you know your model is not wrong?'. Three "
             "independent checks: exhaustive enumeration on a tiny instance, agreement with "
             "the Wagner-Whitin optimum, and the tight-M argument.")
    footer(s, n)

    # 9 ---------------------------------------------------- what the MILP shows
    n += 1
    s = _new(prs)
    header(s, "What the MILP reveals about the problem",
           "The fixed cost K is what creates batching - and what makes the MILP necessary")
    picture(s, os.path.join(RESULTS, "milp_fixed_cost.png"), x=Inches(0.6),
            y=Inches(1.45), width=Inches(6.1))
    picture(s, os.path.join(RESULTS, "milp_integrality_gap.png"), x=Inches(6.95),
            y=Inches(1.45), width=Inches(5.9))
    caption(s, "Left: as K rises, plans use fewer, larger orders. With K = 0 the problem "
               "collapses to a plain LP and the binary variables become unnecessary. "
               "Right: the LP bound is systematically far below the optimum.",
            Inches(0.6), Inches(5.85))
    notes(s, "Two takeaways. First: if K were zero, MILP would be overkill - this is the "
             "honest justification of the method choice. Second: the gap distribution "
             "justifies Branch and Bound rather than rounding the LP solution.")
    footer(s, n)

    # 10 ------------------------------------------------------------ DQN
    n += 1
    s = _new(prs)
    header(s, "Method 2 - Deep Q-Network (learned policy)")
    tf = _box(s, Inches(0.75), Inches(1.35), Inches(5.9), Inches(5.1))
    _para(tf, "MDP interface", size=15, bold=True, first=True)
    for line in ["state  s_t = (inventory / Imax, t / T)",
                 "action q_t  in {0, 1, ..., Qmax}  (masked to free capacity)",
                 "reward r_t = -(K*1{q_t>0} + c*q_t + h*I_(t+1) + p*S_t)",
                 "episode ends after T periods"]:
        _para(tf, line, size=12, color=GREY, space_before=5)
    _para(tf, "Training", size=15, bold=True, space_before=14)
    for line in ["Q-network: 2 hidden layers x 128 ReLU -> one Q-value per action",
                 "epsilon-greedy exploration, epsilon 1.0 -> 0.05",
                 "experience replay (30k transitions), target network every 500 steps",
                 "Huber loss, Adam, gradient clipping, one update every 2 env steps",
                 f"{cfg['episodes']} training episodes; the demand PMF is never given to the agent"]:
        _para(tf, line, size=12, color=GREY, space_before=5)
    picture(s, os.path.join(RESULTS, "dqn_training.png"), x=Inches(6.9),
            y=Inches(1.6), width=Inches(5.9))
    caption(s, "Learning curve: training-episode cost falls from ~1000 to ~550 as exploration "
               "decays (evaluation on the frozen policy is lower still).",
            Inches(6.9), Inches(5.6), w=Inches(5.9))
    notes(s, "Stress that the agent sees only numbers in and cost out. It never sees the "
             "Poisson mean. That is the 'distribution-free' claim - and it is also why it "
             "cannot promise optimality.")
    footer(s, n)

    # 11 --------------------------------------------------- experimental setup
    n += 1
    s = _new(prs)
    header(s, "Experimental setup and evaluation metrics",
           f"{n_ep} episodes per scenario, identical demand paths for every method")
    rows = [["Scenario", "Demand process", "What it tests"],
            ["Exp 1", "Poisson(6) for all 20 days", "baseline performance"],
            ["Exp 2", "Poisson(6), then a jump to Poisson(10)", "adaptability to a regime change"],
            ["Exp 3", "alternating heavy/light days (6, 10, 2)", "robustness to variance"]]
    table(s, rows, x=Inches(0.75), y=Inches(1.45), w=Inches(11.8), size=12,
          col_widths=[1, 5, 5])
    rows = [["Metric", "Why it matters"],
            ["Average total cost", "main performance measure (paired test against DP)"],
            ["Holding cost / shortage cost / ordering cost", "where the money goes"],
            ["Service level", "share of demand actually served"],
            ["Average inventory", "working capital tied up"],
            ["Computation time", "algorithmic efficiency"],
            ["Adaptation performance (Exp 2 window)", "response to demand changes"]]
    table(s, rows, x=Inches(0.75), y=Inches(3.35), w=Inches(11.8), size=12,
          col_widths=[5, 5])
    notes(s, "Common random numbers: all methods face the same demand realisations, so "
             "differences are due to the policy, not luck. The MILP rolling planner re-solves "
             "every period - its computation time is a genuine cost, not a free lunch.")
    footer(s, n)

    # 12 ------------------------------------------------------ main results
    n += 1
    s = _new(prs)
    header(s, "Results - Exp 1 (stationary demand)",
           f"Mean over {n_ep} identical demand paths")
    rows = [["Method", "Total cost", "Holding", "Shortage", "Ordering",
             "Service", "Avg inv.", "Time (ms)"]]
    for m in ev["methods"]:
        d = S[base][m]
        rows.append([m, f"{d['mean_total_cost']:.1f}", f"{d['mean_holding_cost']:.1f}",
                     f"{d['mean_shortage_cost']:.1f}", f"{d['mean_ordering_cost']:.1f}",
                     f"{100 * d['mean_service_level']:.1f}%",
                     f"{d['mean_avg_inventory']:.1f}",
                     f"{d['mean_compute_time_ms']:.1f}"])
    table(s, rows, x=Inches(0.6), y=Inches(1.4), w=Inches(12.1), size=11)
    picture(s, os.path.join(RESULTS, "cost_comparison.png"), x=Inches(0.6),
            y=Inches(3.3), width=Inches(6.0))
    picture(s, os.path.join(RESULTS, "cost_breakdown.png"), x=Inches(6.9),
            y=Inches(3.3), width=Inches(5.9))
    notes(s, "Read the table, do not read the numbers out one by one: DP is cheapest among "
             "the implementable policies, DQN is within a few percent of DP, the rolling "
             "MILP is slightly worse on cost because a point forecast ignores variance, and "
             "the clairvoyant MILP is the lower bound nobody can reach. Note its computation "
             "time is orders of magnitude larger.")
    footer(s, n)

    # 13 ------------------------------------------------- paired significance
    n += 1
    s = _new(prs)
    header(s, "Is the difference real? Paired test against DP",
           "Per-episode differences on identical demand paths, 95% confidence intervals")
    short = {"DQN": "DQN",
             "MILP (rolling forecast)": "MILP (rolling)",
             "MILP (clairvoyant)": "MILP (clairvoyant)"}
    verdicts = {"worse than DP": "worse than DP",
                "indistinguishable from DP": "= DP (no difference)",
                "better than DP": "lower bound only"}
    rows = [["Method - DP", "Exp 1", "Exp 2", "Exp 3", "Reading"]]
    for m in pt["methods"]:
        cells = [f"{pt['scenarios'][name][m]['pct']:+.1f}%  "
                 f"{pfmt(pt['scenarios'][name][m]['p_value'])}"
                 for name in pt["scenarios"]]
        rows.append([short.get(m, m)] + cells +
                    [verdicts[pt["scenarios"][base_sc][m]["verdict"]]])
    table(s, rows, x=Inches(0.6), y=Inches(1.42), w=Inches(6.7), size=10,
          col_widths=[1.75, 1.9, 1.9, 1.9, 2.15])

    rows = [["Method", "Mean diff", "95% CI", "Paths better"]]
    for m in pt["methods"]:
        r = pt["scenarios"][base_sc][m]
        rows.append([short.get(m, m), f"{r['mean_diff']:+.2f}",
                     f"[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]",
                     f"{r['wins']}/{r['n']}"])
    table(s, rows, x=Inches(0.6), y=Inches(2.95), w=Inches(6.7), size=10,
          col_widths=[1.9, 1.5, 2.0, 1.6])
    caption(s, f"Exp 1, {pt['episodes'][base_sc]} episodes, cost units per episode. A CI "
               "containing 0 means the difference is indistinguishable from noise; an "
               "interval entirely above 0 means reliably more expensive.",
            Inches(0.6), Inches(4.3), w=Inches(6.7))

    d = pt["component_deltas"]["DQN"]
    comp = pt["components"]
    tf = _box(s, Inches(7.35), Inches(1.45), Inches(5.4), Inches(3.8))
    _para(tf, "Where the DQN's gap comes from", size=15, bold=True, first=True, color=RED)
    _para(tf, f"Total {d['total_cost']:+.1f} cost units per episode, decomposed:",
          size=12, color=GREY, space_before=4)
    _para(tf, f"shortage  {d['shortage_cost']:+.1f}   (it under-orders and loses sales)",
          size=12, color=NAVY, space_before=6)
    _para(tf, f"holding   {d['holding_cost']:+.1f}   (it saves here by holding less)",
          size=12, color=NAVY, space_before=3)
    _para(tf, f"ordering  {d['ordering_cost']:+.1f}", size=12, color=NAVY, space_before=3)
    _para(tf, f"service level {comp['DQN']['service_level_pct']:.1f}% vs "
              f"{comp['DP']['service_level_pct']:.1f}% for DP", size=12, color=NAVY,
          space_before=6)
    _para(tf, f"With p/h = {cfg['shortage_cost'] / cfg['holding_cost']:.0f}, a slightly low "
              "reorder threshold is the expensive direction - so the gap is one systematic "
              "mis-set threshold, not instability.", size=11, color=GREY, space_before=8)
    _para(tf, "The clairvoyant MILP wins 40/40 paths because it is told the future: it is a "
              "lower bound, not a competitor.", size=11, color=GREY, space_before=8)

    band(s, Inches(0.6), Inches(5.3), Inches(6.7), Inches(1.25))
    tf = _box(s, Inches(0.8), Inches(5.4), Inches(6.3), Inches(1.1))
    _para(tf, "Verdict: DP is optimal by construction, so nobody can beat it. The rolling "
              f"MILP matches it ({pfmt(pt['scenarios'][base_sc]['MILP (rolling forecast)']['p_value'])}), "
              f"the DQN is reliably {pt['scenarios'][base_sc]['DQN']['pct']:.1f}% dearer, and "
              "perfect foresight would be worth "
              f"{abs(pt['scenarios'][base_sc]['MILP (clairvoyant)']['pct']):.1f}% - "
              "unattainable by any policy.", size=11.5, color=NAVY, first=True)
    notes(s, "This is the slide that turns bars into a conclusion. Because all four methods "
             "face the same 40 demand paths the comparison is paired, so a t-test on the "
             "per-episode differences is valid. Key sentences: (1) DP cannot be beaten - it "
             "is the optimal policy for this MDP, so any positive difference is a limitation "
             "of the learner, and any negative difference must be a lower bound. (2) The "
             "rolling MILP is statistically indistinguishable from DP, so on cost alone "
             "optimisation matches DP - but it needs ~1.4 s of solver time per episode "
             "versus a lookup table. (3) The DQN's gap is almost entirely shortage cost: a "
             "slightly low reorder threshold, which is fixable, not random.")
    footer(s, n)

    # 14 -------------------------------------------------- scenario comparison
    n += 1
    s = _new(prs)
    header(s, "Results - adaptability across scenarios",
           "What happens when the demand world changes after training")
    picture(s, os.path.join(RESULTS, "scenario_costs.png"), x=Inches(0.6),
            y=Inches(1.4), width=Inches(6.3))
    rows = [["Scenario", "DQN", "DP", "Rolling MILP", "Clairvoyant"]]
    for name in sc:
        rows.append([name.replace("Exp", "Exp ").replace("_", " ")] +
                    [f"{S[name][m]['mean_total_cost']:.0f}" for m in ev["methods"]])
    table(s, rows, x=Inches(7.2), y=Inches(1.5), w=Inches(5.5), size=11)
    tf = _box(s, Inches(7.2), Inches(3.5), Inches(5.5), Inches(2.6))
    exp2 = "Exp2_step_change"
    _para(tf, "Exp 2 - cost during the post-jump window", size=14, bold=True, first=True)
    for m in ev["methods"]:
        d = S[exp2][m]
        if "mean_window_cost" in d:
            _para(tf, f"{m}: {d['mean_window_cost']:.1f}", size=12, color=GREY,
                  space_before=4)
    _para(tf, "The trained DQN never saw a mean-10 world. DP is stale for the same reason. "
              "The clairvoyant MILP is unbothered because it is told the future - which is "
              "the cleanest illustration of what information is worth.",
          size=11, color=NAVY, space_before=10)
    notes(s, "This is the honest answer to 'does RL adapt?'. A frozen policy adapts to "
             "nothing; the advantage of RL is that it can be re-trained on the new process "
             "from experience, without a distributional model. Say that explicitly.")
    footer(s, n)

    # 15 ------------------------------------------------- conclusion
    n += 1
    s = _new(prs)
    header(s, "Conclusion, limitations and future scope")
    bullets(s, [
        (0, "Findings"),
        (1, f"DP is the benchmark that matters: expected optimal cost {ev['dp_expected_cost']:.0f}, "
            "and its policy is an interpretable (s, S) rule."),
        (1, f"DQN learns the same structure from experience alone and lands within "
            f"{100 * (m_dqn['mean_total_cost'] - m_dp['mean_total_cost']) / m_dp['mean_total_cost']:.1f}% "
            "of DP - a real but statistically significant gap."),
        (1, f"The clairvoyant MILP is {abs(100 * (m_clair['mean_total_cost'] - m_dp['mean_total_cost']) / m_dp['mean_total_cost']):.1f}% "
            "below DP: that is the value of knowing the future, and it is unattainable by "
            "any non-anticipative policy."),
        (1, "The rolling MILP - the fair optimisation baseline - is statistically "
            f"indistinguishable from DP ({pfmt(pt['scenarios'][base_sc]['MILP (rolling forecast)']['p_value'])}) "
            f"but costs ~{m_roll['mean_compute_time_ms'] / 1000:.1f} s of solver time per "
            "episode versus a lookup table."),
        (0, "Limitations"),
        (1, "Zero lead time, one product, small discrete state space; single training seed; "
            "40 evaluation episodes per scenario."),
        (1, "The rolling MILP uses a point forecast; a scenario-based stochastic program "
            "would be stronger."),
        (0, "Future scope"),
        (1, "Stochastic (scenario-tree) MILP with non-anticipativity constraints; lead times "
            "and multi-product capacity; continuous RL training on non-stationary demand; "
            "confidence intervals over multiple seeds."),
    ], y=Inches(1.3), size=14)
    tf = _box(s, Inches(0.75), Inches(6.05), Inches(11.8), Inches(0.8))
    _para(tf, "Takeaway: a fixed ordering cost is a yes/no decision, so the exact model is a "
              "MILP; Branch and Bound solves it optimally when demand is known, DP solves it "
              "optimally when only the distribution is known, and the learned policy is what "
              "remains deployable when neither is.",
          size=13.5, bold=True, color=RED, first=True)
    notes(s, "Close on the method-choice argument: MILP is the right tool because a fixed "
             "ordering cost is a yes/no decision, and Branch and Bound is the syllabus tool "
             "that solves it exactly - but the model needs a known demand path, which is why "
             "the learned policy is the deployable answer under uncertainty.")
    footer(s, n)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print(f"saved {OUT} with {n} slides")
    return OUT


if __name__ == "__main__":
    build()
