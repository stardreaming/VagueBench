"""Exercise real simulator startup, scene restoration, navigation and image/log output."""
from pathlib import Path

from vaguebench.data_tools import load_dataset, validate_data


def smoke_test(args):
    validate_data(data_file=args.data_file, data_root=args.data_root)
    executable = None
    if args.thor_executable:
        import os
        executable = str(Path(args.thor_executable).expanduser().resolve())
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise ValueError('--thor-executable must be an executable file')
    from vaguebench.envs.eb_alfred.EBAlfEnv import EBAlfEnv

    row = load_dataset(args.data_file)[0]
    results = {}
    for condition in ('base', 'fuzzy'):
        env = EBAlfEnv(eval_set=condition, selected_indexes=[0],
                       dataset_path=args.data_file, dataset_root=args.data_root,
                       output_dir=Path(args.output_dir).expanduser().resolve() / condition,
                       x_display=args.x_display, thor_executable=executable)
        try:
            env.reset()
            if env.episode_language_instruction != row[condition]:
                raise ValueError('Simulator instruction does not match the annotation')
            before = env.save_image()
            target = env.episode_data['pddl_params']['object_target']
            action = env.language_skill_set.index(f'find a {target}')
            _, _, _, info = env.step(action, reasoning='Offline simulator smoke test')
            after = env.save_image()
            env.save_episode_log()
            if not info['last_action_success']:
                raise RuntimeError(info['env_feedback'])
            results[condition] = {'episode_id': row['episode_id'],
                                  'action': env.language_skill_set[action],
                                  'action_success': bool(info['last_action_success']),
                                  'images': [before, after]}
        finally:
            env.close()
    return results
