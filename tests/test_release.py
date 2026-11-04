import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from vaguebench.data_tools import DATA_FILE, condition_tasks, load_dataset, validate_data
from vaguebench.evaluator import evaluate, evaluate_episode
from vaguebench.main import main
from vaguebench.results import summarize


class DataTests(unittest.TestCase):
    def test_complete_data_and_paired_views(self):
        result = validate_data()
        self.assertEqual(result, {'pairs': 1303, 'instructions': 2606})
        rows = load_dataset()
        base, fuzzy = condition_tasks(rows, 'base'), condition_tasks(rows, 'fuzzy')
        for b, f, original in zip(base, fuzzy, rows):
            self.assertEqual(b['episode_id'], f['episode_id'])
            self.assertEqual(b['task'], f['task'])
            self.assertEqual(b['repeat_idx'], f['repeat_idx'])
            self.assertEqual(b['instruction'], original['base'])
            self.assertEqual(f['instruction'], original['fuzzy'])

    def test_unsafe_paths_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = load_dataset()
            original_task = rows[0]['task']
            rows[0]['task'] = '../../outside.json'
            path = Path(folder) / 'dataset.json'
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'Nonportable'):
                validate_data(data_file=path)
            rows[0]['task'] = original_task
            rows[1]['episode_id'] = rows[0]['episode_id']
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'episode_id'):
                validate_data(data_file=path)

    def test_data_root_check_reports_missing_trajectories(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Missing or unsafe trajectory'):
                validate_data(data_root=folder)

    def test_cli_rejects_zero_sampling_ratio_before_evaluation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['evaluate', '--data-root', '/missing', '--model-name', 'fake',
                      '--output-dir', '/unused', '--down-sample-ratio', '0'])
        self.assertEqual(error.exception.code, 2)

    def test_cli_rejects_resolution_below_simulator_minimum(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['evaluate', '--data-root', '/missing', '--model-name', 'fake',
                      '--output-dir', '/unused', '--resolution', '299'])
        self.assertEqual(error.exception.code, 2)


class ResultTests(unittest.TestCase):
    def test_summary_pairs_by_id_and_ignores_summary_and_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [('a', 'base', 1), ('b', 'base', 0), ('a', 'fuzzy', 0)]
            for i, (episode, condition, success) in enumerate(records):
                (root / f'{i}_final_res.json').write_text(json.dumps({
                    'status': 'completed', 'episode_id': episode, 'eval_set': condition,
                    'task_success': success, 'task_progress': success, 'fingerprint': 'same'}))
            (root / 'summary.json').write_text('{"task_success": 999}')
            (root / 'a_error.json').write_text('{"status": "error"}')
            first = summarize(root)
            (root / 'summary.json').write_text(json.dumps(first))
            self.assertEqual(summarize(root), first)
            self.assertEqual(first['base']['task_success'], 0.5)
            self.assertEqual(first['paired']['episodes'], 1)
            self.assertEqual(first['paired']['base_only'], 1)
            self.assertEqual(first['paired']['base_success'], 1)

    def test_mixed_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(2):
                (Path(folder) / f'{i}_final_res.json').write_text(json.dumps({
                    'status': 'completed', 'episode_id': str(i), 'eval_set': 'base',
                    'fingerprint': str(i)}))
            with self.assertRaisesRegex(ValueError, 'multiple experiment'):
                summarize(folder)


class FakeEnv:
    starts = 0

    def __init__(self, eval_set='base', selected_indexes=None, dataset_path=None, **kwargs):
        FakeEnv.starts += 1
        self.tasks = [condition_tasks(load_dataset(dataset_path), eval_set)[i] for i in selected_indexes] if dataset_path else [{'instruction': 'test'}]
        self._current_episode_num = 0
        self._max_episode_steps = 30
        self._max_invalid_actions = 10
        self.language_skill_set = ['find a Cup']
        self.closed = False

    def reset(self):
        self.episode_language_instruction = self.tasks[self._current_episode_num]['instruction']
        self._current_episode_num += 1
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._episode_start_time = time.time()
        self.episode_log = []
        return {}

    def save_image(self, obs):
        return 'image.png'

    def step(self, action, reasoning=''):
        self._current_step += 1
        return {}, 1, True, {'last_action_success': True, 'task_success': 1,
                            'task_progress': 1, 'env_step': self._current_step,
                            'action_id': action, 'env_feedback': 'ok'}

    def save_episode_log(self):
        pass

    def close(self):
        self.closed = True


class FakePlanner:
    calls = 0

    def __init__(self, *args, **kwargs):
        self.reset()

    def reset(self):
        self.planner_steps = 0
        self.output_json_error = 0

    def set_actions(self, actions):
        pass

    def act(self, image, instruction):
        FakePlanner.calls += 1
        self.planner_steps += 1
        return [0], 'plan'

    def update_info(self, info):
        pass


class EvaluationTests(unittest.TestCase):
    def test_empty_plan_has_finite_metrics(self):
        planner = FakePlanner()
        planner.act = lambda *args: (-2, 'empty')
        result = evaluate_episode(FakeEnv(), planner)
        self.assertEqual(result['reward'], 0)
        self.assertEqual(result['num_steps'], 0)
        self.assertEqual(result['empty_plan'], 1)
        json.dumps(result, allow_nan=False)

    def test_invalid_model_outputs_stop_at_budget(self):
        planner = FakePlanner()
        planner.act = lambda *args: (-1, 'invalid')
        result = evaluate_episode(FakeEnv(), planner)
        self.assertEqual(result['num_invalid_actions'], 10)
        self.assertEqual(result['num_steps'], 0)

    def test_empty_replan_preserves_last_environment_progress(self):
        env = FakeEnv()
        def partial_step(action, reasoning=''):
            env._current_step += 1
            return {}, 0.5, False, {'last_action_success': True, 'task_success': 0,
                                   'task_progress': 0.5, 'env_step': 1,
                                   'action_id': action, 'env_feedback': 'partly done'}
        env.step = partial_step
        planner = FakePlanner()
        actions = iter([([0], 'first'), (-2, 'stop')])
        planner.act = lambda *args: next(actions)
        result = evaluate_episode(env, planner)
        self.assertEqual(result['task_progress'], 0.5)
        self.assertEqual(result['num_steps'], 1)
        self.assertEqual(result['empty_plan'], 1)

    def test_resume_skips_completed_episodes_and_rejects_changed_configuration(self):
        env_module = types.ModuleType('vaguebench.envs.eb_alfred.EBAlfEnv')
        env_module.EBAlfEnv = FakeEnv
        planner_module = types.ModuleType('vaguebench.planner.vlm_planner')
        planner_module.VLMPlanner = FakePlanner
        modules = {'vaguebench.envs.eb_alfred.EBAlfEnv': env_module,
                   'vaguebench.planner.vlm_planner': planner_module}
        with tempfile.TemporaryDirectory() as folder, patch.dict('sys.modules', modules), patch('vaguebench.evaluator.validate_data'):
            config = dict(data_root='/fake', output_dir=folder, model_name='fake', model_type='remote',
                          data_file=str(DATA_FILE), eval_sets=['base', 'fuzzy'], resume=False,
                          n_shots=10, chat_history=False, language_only=False, env_feedback=True,
                          multistep=0, tp=1, model_options={}, down_sample_ratio=1.0,
                          limit=2, resolution=500, detection_box=False, x_display='1')
            FakePlanner.calls = 0
            FakeEnv.starts = 0
            summary = evaluate(config)
            self.assertEqual(summary['paired']['episodes'], 2)
            self.assertEqual(FakePlanner.calls, 4)
            config['resume'] = True
            self.assertEqual(evaluate(config), summary)
            self.assertEqual(FakePlanner.calls, 4)
            self.assertEqual(FakeEnv.starts, 2)
            config['n_shots'] = 1
            with self.assertRaisesRegex(ValueError, 'differs'):
                evaluate(config)


if __name__ == '__main__':
    unittest.main()
