"""Command line entry point; data commands require only Python's standard library."""
import argparse
import json
import logging
import os
from pathlib import Path

from vaguebench.data_tools import DATA_FILE, validate_data
from vaguebench.results import summarize


def parser():
    root = argparse.ArgumentParser(description='VagueBench: paired ALFRED base/fuzzy evaluation')
    commands = root.add_subparsers(dest='command', required=True)
    validation = commands.add_parser('validate-data', help='Check the merged canonical annotation dataset')
    validation.add_argument('--data-root', help='Optional ALFRED json_2.1.0 directory; also check trajectory files')
    validation.add_argument('--data-file', default=str(DATA_FILE))
    summary = commands.add_parser('summarize', help='Summarize one experiment output directory')
    summary.add_argument('directory')
    smoke = commands.add_parser('smoke-test', help='Reset both conditions and execute one simulator action, without a model API')
    smoke.add_argument('--data-root', required=True)
    smoke.add_argument('--data-file', default=str(DATA_FILE))
    smoke.add_argument('--output-dir', required=True)
    smoke.add_argument('--x-display', default='1')
    smoke.add_argument('--thor-executable')
    evaluation = commands.add_parser('evaluate', help='Evaluate ALFRED through an API or optional local backend')
    evaluation.add_argument('--data-root', required=True, help='ALFRED json_2.1.0 directory')
    evaluation.add_argument('--model-name', required=True)
    evaluation.add_argument('--model-type', choices=['remote', 'local', 'custom'], default='remote')
    evaluation.add_argument('--provider', choices=['auto', 'openai', 'anthropic', 'gemini', 'dashscope'], default='auto')
    evaluation.add_argument('--base-url', default=None)
    evaluation.add_argument('--api-key-env', default=None, help='Environment variable NAME; never put a key here')
    evaluation.add_argument('--response-format', choices=['json_schema', 'json_object', 'text'], default='json_schema')
    evaluation.add_argument('--data-file', default=str(DATA_FILE))
    evaluation.add_argument('--eval-sets', nargs='+', choices=['base', 'fuzzy'], default=['base', 'fuzzy'])
    evaluation.add_argument('--output-dir', required=True)
    evaluation.add_argument('--resume', action='store_true')
    evaluation.add_argument('--n-shots', type=int, default=10)
    evaluation.add_argument('--limit', type=int, default=None)
    evaluation.add_argument('--down-sample-ratio', type=float, default=1.0)
    evaluation.add_argument('--resolution', type=int, default=500)
    evaluation.add_argument('--x-display', default='1')
    evaluation.add_argument('--thor-executable', help='Optional existing AI2-THOR 2.1.0 executable; avoids downloading Unity')
    evaluation.add_argument('--language-only', action='store_true')
    evaluation.add_argument('--chat-history', action='store_true')
    evaluation.add_argument('--detection-box', action='store_true')
    evaluation.add_argument('--multistep', type=int, default=0)
    evaluation.add_argument('--no-env-feedback', dest='env_feedback', action='store_false')
    evaluation.add_argument('--tp', type=int, default=1)
    return root


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    if args.command == 'validate-data':
        result = validate_data(data_file=args.data_file, data_root=args.data_root)
    elif args.command == 'summarize':
        result = summarize(args.directory)
    elif args.command == 'smoke-test':
        from vaguebench.smoke import smoke_test
        result = smoke_test(args)
    else:
        if not 0 < args.down_sample_ratio <= 1:
            argument_parser.error('--down-sample-ratio must be in (0, 1]')
        if args.limit is not None and args.limit <= 0:
            argument_parser.error('--limit must be positive')
        if args.resolution < 300 or args.tp <= 0 or min(args.n_shots, args.multistep) < 0:
            argument_parser.error('Invalid resolution, tp, n-shots, or multistep')
        if args.multistep and (args.chat_history or args.language_only):
            argument_parser.error('--multistep cannot be combined with --chat-history or --language-only')
        config = vars(args).copy()
        config['data_root'] = str(Path(args.data_root).expanduser().resolve())
        config['data_file'] = str(Path(args.data_file).expanduser().resolve())
        if args.thor_executable:
            executable = Path(args.thor_executable).expanduser().resolve()
            if not executable.is_file() or not os.access(executable, os.X_OK):
                argument_parser.error('--thor-executable must be an executable file')
            config['thor_executable'] = str(executable)
        config['eval_sets'] = list(dict.fromkeys(args.eval_sets))
        config['model_options'] = {key: config.pop(key) for key in ['provider', 'base_url', 'api_key_env', 'response_format']}
        # Capture the effective endpoint without reading or persisting credentials.
        config['model_options']['base_url'] = config['model_options']['base_url'] or os.environ.get('VAGUEBENCH_BASE_URL')
        from vaguebench.evaluator import evaluate
        result = evaluate(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
