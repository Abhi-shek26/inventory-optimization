import unittest
from dataclasses import replace

from algorithms.branch_and_bound import branch_and_bound
from algorithms.dynamic_programming import DynamicProgrammingSolver
from algorithms.lot_sizing import wagner_whitin_cost
from algorithms.milp_optimizer import solve_milp, solve_rolling_milp
from config import Config
from environment import InventoryEnv
from experiments.milp_analysis import enumerate_setup_patterns
from utils.simulation import replay_plan, run_episode


class TestInventoryEnvironment(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(horizon=3, max_inventory=10, max_order=5,
                          demand_values=(0, 1, 2), demand_lambda=1.0,
                          fixed_order_cost=25.0)
        self.env = InventoryEnv(self.cfg, demand_pmf=[.2, .5, .3], seed=1)

    def test_inventory_transition_and_cost(self):
        self.env.reset()
        _, reward, _, info = self.env.step(3, demand=2)
        self.assertEqual(info["ending_inventory"], 1)
        self.assertAlmostEqual(
            -reward, 25.0 + 3 * self.cfg.order_cost + self.cfg.holding_cost)

    def test_fixed_cost_is_not_charged_for_zero_order(self):
        self.env.reset()
        _, reward, _, info = self.env.step(0, demand=0)
        self.assertEqual(info["fixed_cost"], 0.0)
        self.assertEqual(-reward, 0.0)

    def test_action_capacity(self):
        self.env.inventory = 10
        self.assertEqual(list(self.env.valid_actions()), [0])

    def test_dp_policy_shape(self):
        solver = DynamicProgrammingSolver(self.cfg, self.env.values, self.env.pmf)
        value, policy = solver.solve()
        self.assertGreaterEqual(value, 0)
        self.assertEqual(policy.shape, (self.cfg.horizon, self.cfg.max_inventory + 1))

    def test_dp_never_orders_without_need(self):
        solver = DynamicProgrammingSolver(self.cfg, self.env.values, self.env.pmf)
        solver.solve()
        # Full warehouse must imply no ordering.
        self.assertEqual(solver.act(0, self.cfg.max_inventory), 0)


class TestMILP(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(horizon=4, max_order=12, max_inventory=12,
                          demand_values=(0, 1, 2, 3, 4, 5, 6, 7, 8),
                          fixed_order_cost=25.0, order_cost=2.0,
                          holding_cost=0.5, shortage_cost=8.0)
        self.demand = [4, 7, 3, 6]

    def test_milp_matches_brute_force_enumeration(self):
        rows, best = enumerate_setup_patterns(self.demand, self.cfg)
        milp = solve_milp(self.demand, self.cfg)
        self.assertAlmostEqual(milp["cost"], best["cost"], places=6)

    def test_milp_matches_wagner_whitin_when_uncapacitated(self):
        off = replace(self.cfg, max_order=10_000, max_inventory=10_000,
                      shortage_cost=1e6)
        for demand in ([6, 5, 7, 6, 8], [3, 3, 3, 3, 3], [9, 0, 9, 0, 9]):
            milp = solve_milp(demand, off)
            ww = wagner_whitin_cost(demand, off)
            self.assertAlmostEqual(milp["cost"], ww["cost"], places=6)

    def test_big_m_forces_zero_quantity_when_setup_is_off(self):
        out = solve_milp(self.demand, self.cfg)
        for q, y in zip(out["orders"], out["setups"]):
            if y == 0:
                self.assertEqual(q, 0)
            else:
                self.assertLessEqual(q, self.cfg.setup_big_m())
                self.assertGreaterEqual(q, 1)

    def test_tight_big_m_is_not_larger_than_capacity_or_order_limit(self):
        self.assertEqual(self.cfg.setup_big_m(),
                         min(self.cfg.max_order, self.cfg.max_inventory))

    def test_branch_and_bound_matches_cbc(self):
        milp = solve_milp(self.demand, self.cfg)
        bnb = branch_and_bound(self.demand, self.cfg)
        self.assertAlmostEqual(bnb["cost"], milp["cost"], places=6)
        self.assertTrue(bnb["optimal"])

    def test_lp_relaxation_is_a_lower_bound(self):
        bound = solve_milp(self.demand, self.cfg, relax=True)
        exact = solve_milp(self.demand, self.cfg)
        self.assertLessEqual(bound["cost"], exact["cost"] + 1e-6)

    def test_plan_replay_cost_matches_model_objective(self):
        out = solve_milp(self.demand, self.cfg)
        replay = replay_plan(out["orders"], self.demand, self.cfg)
        self.assertAlmostEqual(replay["total_cost"], out["cost"], places=6)

    def test_rolling_forecast_is_implementable(self):
        out = solve_rolling_milp(self.demand, self.cfg)
        self.assertEqual(len(out["orders"]), len(self.demand))
        self.assertGreater(out["cost"], 0)


class TestSimulationAgreement(unittest.TestCase):
    def test_run_episode_matches_environment(self):
        cfg = Config(horizon=5, max_inventory=10, max_order=5,
                     demand_values=(0, 1, 2, 3, 4, 5, 6), demand_lambda=3.0)
        actions = [4, 0, 5, 2, 0]
        demands = [3, 6, 1, 4, 2]
        it = iter(actions)
        sim = run_episode(lambda t, i: next(it), demands, cfg)

        env = InventoryEnv(cfg, seed=0)
        env.reset()
        total = 0.0
        for q, d in zip(actions, demands):
            _, reward, _, _ = env.step(q, demand=d)
            total -= reward
        self.assertAlmostEqual(sim["total_cost"], total, places=6)


if __name__ == "__main__":
    unittest.main()
