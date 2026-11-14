# Team sync contract — MILP part ⇄ RL part

This repo holds the **MILP part**. The RL part lives in the same repository
structure inside the other member's working copy. This page is the agreement that
keeps the two halves comparable. If a number is ever disputed, this file decides.

## 1. The rule

**One model, one simulator, one demand stream, one metrics function — two
policies.** Nobody owns their own copy of the problem. Each member owns only the
code that decides an order quantity, and plugs it into the shared harness.

## 2. The interface (the only function either of us implements)

```python
policy(t: int, inventory: int) -> q: int        # units to order this period
```

The shared harness (`utils/simulation.py::run_episode`) does everything else:
clipping to `min(Qmax, Imax - inventory)`, the physics, and the cost

```
period_cost = K*1{q>0} + c*q + h*ending_inventory + p*lost_sales
```

Consequence: nobody can accidentally give their method a cheaper cost function,
a bigger action space or a different demand path.

| Part | Files owned by that member |
|---|---|
| **MILP (mine)** | `algorithms/milp_optimizer.py`, `algorithms/branch_and_bound.py`, `experiments/milp_analysis.py` |
| **RL (theirs)** | `algorithms/dqn_agent.py`, `experiments/train_dqn.py` (+ their own analysis script) |
| **Shared — identical in both repos, never edited unilaterally** | `config.py`, `environment.py`, `utils/simulation.py`, `utils/visualization.py`, `experiments/demand_scenarios.py`, `experiments/evaluate.py`, `experiments/statistics.py` |

`experiments/evaluate.py` is the single comparison harness. It imports the RL
module lazily, so the same file produces four methods where the RL files exist and
three methods (`DP`, `MILP rolling`, `MILP clairvoyant`) where they don't. That is
why each member's repo can run the same command and get a consistent table.

## 3. Frozen parameters (change = both re-run, both sign)

| Parameter | Value | Where |
|---|---|---|
| horizon `T` | 20 | `config.py` |
| warehouse `Imax` | 40 | `config.py` |
| order cap `Qmax` | 30 | `config.py` |
| fixed order cost `K` | 25.0 | `config.py` |
| unit cost `c` | 2.0 | `config.py` |
| holding `h` | 0.5 | `config.py` |
| lost-sale penalty `p` | 8.0 | `config.py` |
| demand support | integers 0..12 | `config.py` |
| demand process | truncated Poisson, λ = 6 (truncation keeps 99.1% of mass) | `environment.py` |
| evaluation episodes | 40 per scenario, seeds `seed + 10000 + ep` | `experiments/evaluate.py` |
| rolling MILP forecast | point forecast = demand mean, lookahead 5 | `experiments/evaluate.py` |
| training seed | 42 | `config.py` |

`Imax` is baked into the RL state normalisation and into both action masks — treat
it as untouchable once a model has been trained.

## 4. Information protocol (the fairness clause — quote it verbatim)

| Method | Sees at decision time | Implementable? |
|---|---|---|
| DQN | current state only (inventory, period) | yes |
| DP | the demand *distribution* (truncated Poisson) | yes, if the distribution is known |
| MILP rolling | a 5-period point forecast, re-planned every period | yes |
| MILP clairvoyant | the entire realised demand path | **no** — a lower bound only |

Never compare the clairvoyant MILP with the policies as if it were achievable; its
value is the *expected value of perfect information*.

## 5. Classifying a change before making it

**(A) Free — no coordination needed.** Network size, learning rate, ε schedule,
replay size, training episodes, seeds for training; state features added *inside*
the RL wrapper; extra MILP valid inequalities, solver options, extra diagnostic
plots. None of these touch the shared primitives, so existing results stay valid.

**(B) Requires a joint re-run and both signatures.** Any change to the frozen
table in §3, the lost-sales rule, the reward definition, the evaluation seed rule,
the episode count, or the forecast lookahead. If the RL side needs a different
`K`, that is a *named variant* (`Config(fixed_order_cost=5)`), and the MILP/DP
rows are regenerated for that same variant — never reported as if it were the
default instance.

## 6. Numbers of record

Produced by exactly one command on one machine, then committed and tagged:

```bash
python -m experiments.evaluate        # -> results/evaluation_episodes.csv, evaluation_summary.json, summary.csv
python -m experiments.statistics      # -> results/paired_tests.json, paired_tests.csv
```

`evaluation_summary.json` embeds the full config, so every published table is
traceable to the parameters that produced it. Do not quote numbers from a run
nobody else can reproduce.

## 7. Git workflow

- One repo. Members edit their own files, never the shared ones without a message.
- Tag the frozen state after the first successful joint run: `git tag protocol-v1`.
- Never commit `__pycache__/` or `.DS_Store` (already in `.gitignore`).
- `results/` is generated, not committed (already in `.gitignore`). It is recreated
  by the three commands in `RUNBOOK.md`; anyone can rebuild it from a clean clone.
- After any model change: both re-run `evaluate`, and whoever changed it posts the
  diff of `results/summary.csv`. A non-empty diff that nobody announced = drift;
  stop and find it before continuing.
- Publish the final state as a tag: `git tag results-v1` on the commit that produced
  the reported numbers, and cite the tag in the report. The script `results/` itself
  is not committed — it is regenerated deterministically from that commit
  (`seed = 42`, fixed evaluation seeds), so every table maps to an exact code state.

## 8. Final integration checklist

1. Both repos: `git pull`, then reproduce the same DP cost on the same 40 paths
   (`487.8` on Exp 1 with the frozen parameters). If the number differs, the
   environments differ — fix that before comparing anything (check the PuLP
   version: this model needs `<4`).
2. One member runs the joint pipeline (with the RL files present) and pushes
   the tagged results.
3. Split the write-up: the MILP sections (formulation, big-M, B&B,
   validation, paired test) are mine; RL training and policy analysis are
   theirs; the conclusion is joint.
