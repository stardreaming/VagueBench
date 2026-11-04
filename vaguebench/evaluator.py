"""ALFRED evaluation using matched base/fuzzy instructions."""
import hashlib
import json
from pathlib import Path
import time

from vaguebench.data_tools import DATA_FILE, load_dataset, validate_data
from vaguebench.logging_utils import logger
from vaguebench.results import summarize


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def evaluate_episode(env, planner):
    obs = env.reset()
    image = env.save_image(obs)
    planner.reset()
    planner.set_actions(env.language_skill_set)
    rewards = []
    invalid_actions = 0
    empty_plan = 0
    info = {'task_success': 0.0, 'task_progress': 0.0, 'env_step': 0}
    done = False
    while not done:
        action, reasoning = planner.act(image, env.episode_language_instruction)
        if action == -2:
            empty_plan = 1
            break
        if action == -1:
            invalid_actions += 1
            env._cur_invalid_actions += 1
            rewards.append(-1)
            env.episode_log.append({'action_id': -1, 'action_description': 'invalid action', 'reasoning': reasoning})
            if env._cur_invalid_actions >= env._max_invalid_actions:
                break
            continue
        actions = action if isinstance(action, list) else [action]
        for single in actions[:env._max_episode_steps - env._current_step]:
            obs, reward, done, info = env.step(single, reasoning=reasoning)
            rewards.append(float(reward))
            invalid_actions += int(not info['last_action_success'])
            planner.update_info(info)
            image = env.save_image(obs)
            if done or not info['last_action_success']:
                break
    result = {
        'instruction': env.episode_language_instruction,
        'reward': sum(rewards) / len(rewards) if rewards else 0.0,
        'task_success': float(info['task_success']),
        'task_progress': float(info['task_progress']),
        'num_steps': int(info['env_step']), 'planner_steps': planner.planner_steps,
        'planner_output_error': planner.output_json_error,
        'num_invalid_actions': invalid_actions,
        'num_invalid_action_ratio': invalid_actions / info['env_step'] if info['env_step'] else 0.0,
        'empty_plan': empty_plan,
        'episode_elapsed_seconds': time.time() - env._episode_start_time,
    }
    env.save_episode_log()
    return result


def evaluate(config):
    # Validate paths before importing simulator dependencies or creating API clients.
    validate_data(data_file=config['data_file'], data_root=config['data_root'])
    from vaguebench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
    from vaguebench.planner.vlm_planner import VLMPlanner
    from vaguebench.prompts.system_prompts import alfred_system_prompt

    output = Path(config['output_dir']).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_file = Path(__file__).parent / 'prompts/alfred_examples.json'
    examples = json.loads(examples_file.read_text(encoding='utf-8'))
    identity = {k: v for k, v in config.items() if k not in {'output_dir', 'resume', 'command'}}
    identity['prompt_sha256'] = hashlib.sha256(alfred_system_prompt.encode()).hexdigest()
    identity['examples_sha256'] = hashlib.sha256(examples_file.read_bytes()).hexdigest()
    identity['dataset_sha256'] = hashlib.sha256(Path(config['data_file']).read_bytes()).hexdigest()
    identity['format_version'] = 1
    source_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.rglob('*.py')):
        source_hash.update(str(source.relative_to(Path(__file__).parent)).encode())
        source_hash.update(source.read_bytes())
    identity['code_sha256'] = source_hash.hexdigest()
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config_file = output / 'run_config.json'
    if config_file.exists():
        previous = json.loads(config_file.read_text(encoding='utf-8'))
        if not config['resume']:
            raise ValueError('Output already has a run; use --resume or choose a new --output-dir')
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Resume configuration/data/prompt differs from the saved run')
    elif any(output.iterdir()):
        raise ValueError('Output directory must be empty for a new run')
    write_json(config_file, {'fingerprint': fingerprint, 'config': identity})
    planner = VLMPlanner(
        config['model_name'], config['model_type'], [], alfred_system_prompt, examples,
        n_shot=config['n_shots'], chat_history=config['chat_history'],
        language_only=config['language_only'], use_feedback=config['env_feedback'],
        multistep=config['multistep'], tp=config['tp'],
        kwargs={'model': config['model_options']})

    data_file = Path(config['data_file'])
    data = load_dataset(data_file)
    stride = max(1, round(1 / config['down_sample_ratio']))
    indexes = list(range(0, len(data), stride))
    if config['limit'] is not None:
        indexes = indexes[:config['limit']]
    for condition in config['eval_sets']:
        folder = output / condition
        expected = [data[i]['episode_id'] for i in indexes]
        # Completed conditions can be resumed without starting Unity.
        if all((folder / 'results' / f'{eid}_final_res.json').exists() for eid in expected):
            for eid in expected:
                saved = json.loads((folder / 'results' / f'{eid}_final_res.json').read_text())
                if saved.get('fingerprint') != fingerprint or saved.get('status') != 'completed':
                    raise ValueError('Saved episode does not belong to this completed run')
            continue
        env = None
        try:
            env = EBAlfEnv(eval_set=condition, selected_indexes=indexes,
                dataset_path=data_file, dataset_root=config['data_root'], output_dir=folder,
                resolution=config['resolution'], detection_box=config['detection_box'],
                x_display=config['x_display'], thor_executable=config.get('thor_executable'))
            for index in indexes:
                task = data[index]
                path = folder / 'results' / f'{task["episode_id"]}_final_res.json'
                if path.exists():
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    if saved.get('fingerprint') != fingerprint or saved.get('status') != 'completed':
                        raise ValueError('Saved episode does not belong to this completed run')
                    env._current_episode_num += 1
                    continue
                logger.info('Evaluating %s / %s', task['episode_id'], condition)
                try:
                    result = evaluate_episode(env, planner)
                except Exception as error:
                    write_json(path.with_name(task['episode_id'] + '_error.json'), {
                        'status': 'error', 'episode_id': task['episode_id'],
                        'eval_set': condition, 'error_type': type(error).__name__,
                        'fingerprint': fingerprint})
                    raise
                result.update(status='completed', episode_id=task['episode_id'],
                              eval_set=condition, task=task['task'],
                              repeat_idx=task['repeat_idx'], fingerprint=fingerprint)
                write_json(path, result)
        finally:
            if env is not None:
                env.close()
        write_json(folder / 'summary.json', summarize(folder))
    summary = summarize(output)
    write_json(output / 'summary.json', summary)
    return summary
