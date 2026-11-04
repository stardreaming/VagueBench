"""Runtime regressions; skipped when the optional evaluation dependencies are absent."""
import importlib.util
import unittest
from unittest.mock import patch
import types


RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None
                        for name in ('numpy', 'gym', 'cv2', 'ai2thor', 'scipy', 'networkx', 'PIL'))


@unittest.skipUnless(RUNTIME_AVAILABLE, 'Requires the eval extra (or embench2)')
class RuntimeTests(unittest.TestCase):
    def test_scene_restoration_clears_previous_episode_state(self):
        from vaguebench.envs.eb_alfred.thor_connector import ThorConnector, ThorEnv
        connector = ThorConnector.__new__(ThorConnector)
        connector.cur_receptacle = 'Fridge'
        connector.sliced = True
        connector.put_count_dict = {'Fridge': 5}
        with patch.object(ThorEnv, 'restore_scene'), patch.object(
                ThorConnector, 'get_reachable_positions', return_value=([], None)):
            connector.restore_scene([], [], False)
        self.assertIsNone(connector.cur_receptacle)
        self.assertFalse(connector.sliced)
        self.assertEqual(connector.put_count_dict, {})

    def test_navigation_to_explicit_id_does_not_require_a_name_match(self):
        from vaguebench.envs.eb_alfred.thor_connector import ThorConnector
        connector = ThorConnector.__new__(ThorConnector)
        connector.last_event = types.SimpleNamespace(metadata={'objects': []})
        with patch.object(ThorConnector, 'get_obj_id_from_name', return_value=(None, None)):
            message = connector.nav_obj('Cabinet|1|2|3')
        self.assertIn('Cannot find', message)

    def test_text_and_json_object_backends_receive_output_schema(self):
        from vaguebench.planner.vlm_planner import VLMPlanner
        from vaguebench.prompts.system_prompts import alfred_system_prompt
        for response_format in ('text', 'json_object'):
            model = types.SimpleNamespace(response_format=response_format, provider='openai',
                                          respond=lambda messages: '{"executable_plan": []}')
            with patch('vaguebench.planner.vlm_planner.RemoteModel', return_value=model):
                planner = VLMPlanner('example', 'remote', ['find a Cup'],
                                     alfred_system_prompt, [], language_only=True)
                planner.reset()
                planner.act('unused.png', 'Find a cup')
            prompt = planner.episode_messages[0]['content'][0]['text']
            self.assertIn('executable_plan', prompt)
            self.assertIn('action_id', prompt)

    def test_image_annotation_supports_installed_pillow(self):
        import numpy as np
        from PIL import ImageFont
        from vaguebench.envs.eb_alfred.thor_connector import ThorConnector
        connector = ThorConnector.__new__(ThorConnector)
        connector.last_event = types.SimpleNamespace(frame=np.zeros((100, 100, 3), dtype=np.uint8))
        connector.font = ImageFont.load_default()
        image = connector.write_step_on_img(True, 0, {'action': 'find', 'success': False, 'message': 'missing'})
        self.assertEqual(image.size, (100, 100))
