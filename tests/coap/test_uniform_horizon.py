"""Uniform per-chunk horizon boundaries, inclusive sampling and RNG isolation."""
import random
import unittest
from collections import deque
from unittest.mock import Mock, patch
import numpy as np
import torch
from siglip_dynamics.random_replan import Config, Evaluation, RandomSettings, RandomReplanController, after_step, load_config

class UniformHorizonTests(unittest.TestCase):
    def settings(self,**updates):
        return RandomSettings(strategy='uniform_horizon',warmup_steps=0,cooldown_steps=0,**updates)

    def test_draw_once_per_chunk_inclusive_endpoints_and_discard_only_unexecuted(self):
        controller=RandomReplanController(self.settings())
        rng=Mock()
        rng.integers.side_effect=[10,50,25,40]
        with patch('siglip_dynamics.random_replan.np.random.default_rng',return_value=rng):
            controller.reset('easy:0',1000)
        queue,raw=deque(),deque()
        starts=[];executed=[];events=[]
        for t in range(90):
            if not queue:
                starts.append(t)
                queue.extend([(t,i) for i in range(50)])
                raw.extend([(t,i) for i in range(50)])
            executed.append(queue.popleft());raw.popleft()
            events.append(after_step(controller,t+1,queue,raw,done=t==89))
        self.assertEqual(starts,[0,10,60,85])
        self.assertEqual([e['completed_step'] for e in events if e['trigger']],[10,85])
        self.assertEqual([e['discarded_actions'] for e in events if e['trigger']],[40,25])
        self.assertFalse(events[59]['trigger'])  # Horizon 50 is a natural boundary.
        self.assertFalse(events[-1]['trigger']) # Never replan a terminal episode.
        self.assertEqual([args.args for args in rng.integers.call_args_list],[(10,51)]*4)
        for start,length in zip(starts,[10,50,25,5]):
            self.assertEqual(executed[start:start+length],[(start,i) for i in range(length)])

    def test_reproducible_stream_preserves_global_rngs_and_rejects_invalid_configs(self):
        numpy_state=np.random.get_state();torch_state=torch.random.get_rng_state().clone();python_state=random.getstate()
        def episode(task,seed):
            c=RandomReplanController(self.settings());c.reset(task,seed)
            queue,raw=deque(),deque()
            for t in range(250):
                if not queue:queue.extend(range(50));raw.extend(range(50))
                queue.popleft();raw.popleft();after_step(c,t+1,queue,raw,done=t==249)
            return [e['planned_horizon'] for e in c.traces if e['chunk_action_index']==1]
        first=episode('easy:0',1000)
        self.assertEqual(first,episode('easy:0',1000))
        self.assertNotEqual(first,episode('hard:0',1000))
        self.assertTrue(all(10<=h<=50 for h in first))
        self.assertEqual(numpy_state[0],np.random.get_state()[0])
        np.testing.assert_array_equal(numpy_state[1],np.random.get_state()[1])
        self.assertEqual(numpy_state[2:],np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_state,torch.random.get_rng_state()))
        self.assertEqual(python_state,random.getstate())
        for changes in [dict(min_horizon=0),dict(min_horizon=51),dict(max_horizon=10.5)]:
            with self.assertRaises(ValueError):self.settings(**changes).validate()
        with self.assertRaises(ValueError):Config(random=self.settings(),evaluation=Evaluation(n_action_steps=10)).validate()
        cfg=load_config('configs/baselines/metaworld_pi05_uniform_10_50.yaml')
        self.assertEqual((cfg.evaluation.n_action_steps,cfg.evaluation.chunk_size),(50,50))

if __name__=='__main__':unittest.main()
