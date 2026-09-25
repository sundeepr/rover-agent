"""Hardware-free behavior checks: python3 experimental/test_relative_move.py."""
import ast
import asyncio
import collections
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import types
import unittest


def load_server(path):
    # Load control logic without opening hardware or importing rover drivers.
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             or isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id != 'ROVER_AGENT_ROOT' for t in n.targets)]
    ns = dict(asyncio=asyncio, collections=collections, dataclass=dataclass, json=json,
              math=math, time=time, serial=types.SimpleNamespace(Serial=object, SerialException=OSError),
              AtlasController=object)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    ns['render_dashboard'] = lambda state: None
    return ns


class Serial:
    def __init__(self):
        self.commands = []
    def write(self, data):
        self.commands.append(json.loads(data))


class StartupHomeTest(unittest.TestCase):
    def test_startup_commands_configured_home_instead_of_firmware_home(self):
        ns = load_server(Path(__file__).with_name('roarm_socket_server.py'))
        ser = Serial()
        feedback = iter([{'x': 250, 'y': 0, 'z': 0}, {'x': 40, 'y': 0, 'z': 150}])
        ns['request_feedback'] = lambda port: next(feedback)
        state = ns['initialize_arm']('right', ser)
        expected = json.loads(ns['joint_command'](ns['EeTarget'](40, 0, 150, 2.715)))
        expected['spd'] = ns['HOME_JOINT_SPEED']
        self.assertEqual(ser.commands, [expected])
        self.assertEqual(state.control_anchor_target, ns['EeTarget'](40, 0, 150, 2.715))
        self.assertFalse(state.gripper_closed)

    def test_missing_feedback_keeps_arm_connected_at_commanded_target(self):
        ns = load_server(Path(__file__).with_name('roarm_socket_server.py'))
        ns['HOME_FEEDBACK_TIMEOUT_S'] = 0
        ser = Serial()
        state = ns['initialize_arm']('right', ser)
        self.assertEqual(state.target, ns['EeTarget'](40, 0, 150, 2.715))
        self.assertFalse(state.gripper_closed)
        self.assertEqual(len(ser.commands), 1)

    def test_off_target_feedback_becomes_control_anchor(self):
        ns = load_server(Path(__file__).with_name('roarm_socket_server.py'))
        clock = iter([0, 0, 10])
        ns['time'] = types.SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None)
        ns['request_feedback'] = lambda ser: {'x': 44, 'y': 1, 'z': 155, 't': 2.715}
        state = ns['initialize_arm']('left', Serial())
        self.assertEqual(state.target, ns['EeTarget'](44, 1, 155, 2.715))
        self.assertEqual(state.control_anchor_target, state.target)


class RelativeMoveTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = load_server(Path(__file__).with_name('roarm_socket_server.py'))
        self.state = self.ns['TeleopState']('right')
        self.ser = Serial()
        self.replies = []
        self.ns['request_feedback'] = lambda ser: self.state.target.__dict__.copy()

    def move(self, request_id='one', delta=None):
        self.ns['handle_relative_move']({'request_id': request_id,
            'delta_mm': delta or {'x': 0, 'y': 0, 'z': 30}, 'duration_ms': 100},
            self.state, self.ser, self.replies.append)

    async def test_move_preserves_gripper_and_blocks_manual_motion(self):
        angle = self.state.target.t
        self.move()
        task = self.state.move_task
        self.ns['handle_teleop_message']({'control_active': True, 'delta': {'z': .2}}, self.state, self.ser)
        self.assertEqual(self.ser.commands, [])
        await task
        self.assertEqual((self.state.target.x, self.state.target.y, self.state.target.z), (40, 0, 180))
        self.assertEqual(self.state.target.t, angle)
        self.assertTrue(all(c['T'] == 102 for c in self.ser.commands))
        self.assertEqual([r['status'] for r in self.replies], ['accepted', 'completed'])
        self.assertFalse(self.state.control_active)
        self.assertEqual(self.state.control_anchor_target.z, 180)

    async def test_duplicate_does_not_repeat(self):
        self.move()
        task = self.state.move_task
        self.move()
        self.assertIs(task, self.state.move_task)
        await task
        self.move()
        self.assertIsNone(self.state.move_task)
        self.assertEqual(self.state.target.z, 180)
        self.assertEqual(self.replies[-1]['status'], 'completed')

    async def test_cancel_stops_remaining_steps(self):
        self.move()
        task = self.state.move_task
        await asyncio.sleep(.02)
        self.ns['cancel_relative_move'](self.state)
        count = len(self.ser.commands)
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(count, len(self.ser.commands))
        self.assertLess(self.state.target.z, 180)
        self.assertEqual(self.replies[-1]['status'], 'cancelled')

    async def test_invalid_and_unreachable_moves_rejected(self):
        for delta in [{'x': 0, 'y': 0, 'z': 101}, {'x': float('nan'), 'y': 0, 'z': 0}]:
            self.move(delta=delta)
            self.assertEqual(self.replies[-1]['status'], 'rejected')
        self.state.target.z = self.ns['MAX_Z_MM']
        self.move()
        self.assertEqual(self.replies[-1]['status'], 'rejected')
        self.assertEqual(self.ser.commands, [])

    async def test_feedback_timeout_reports_failure(self):
        self.ns['MOVE_FEEDBACK_TIMEOUT_S'] = 0
        self.move()
        await self.state.move_task
        self.assertEqual(self.replies[-1]['status'], 'failed')
        self.assertTrue(self.state.move_failed)

    async def test_raw_transport_returns_completion(self):
        reader = asyncio.StreamReader()
        class Writer:
            def __init__(self): self.data = []
            def get_extra_info(self, name): return 'test-client'
            def write(self, data): self.data.append(json.loads(data))
            def close(self): pass
        writer = Writer()
        client = asyncio.create_task(self.ns['handle_raw_client'](
            reader, writer, {'right': self.ser}, {'right': self.state},
            None, self.ns['RoverDriveState']()))
        reader.feed_data((json.dumps({'type': 'arm_move_relative', 'arm': 'right',
            'request_id': 'tcp-test', 'delta_mm': {'x': 0, 'y': 0, 'z': 30},
            'duration_ms': 100}) + '\n').encode())
        await asyncio.sleep(.01)
        await self.state.move_task
        reader.feed_eof()
        await client
        self.assertEqual([r['status'] for r in writer.data], ['accepted', 'completed'])
        self.assertTrue(all(r['request_id'] == 'tcp-test' for r in writer.data))

    async def test_generic_x_move_preserves_hand(self):
        self.state.target.t = 2.8
        self.move(delta={'x': 10, 'y': 0, 'z': 0})
        await self.state.move_task
        self.assertEqual(self.state.target.x, 50)
        self.assertEqual(self.state.target.z, 150)
        self.assertEqual(self.state.target.t, 2.8)


if __name__ == '__main__':
    unittest.main()
