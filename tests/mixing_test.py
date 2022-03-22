from itertools import *
import unittest

from sprucfluo.mixing import sample


class SampleTest(unittest.TestCase):
    def test_sample_with_infinite_iter(self):
        a = repeat('a')
        b = repeat('b')
        datasets = {'a': a, 'b': b }
        weights = {'a': 1, 'b': 1}

        samples = list(islice(sample(datasets, weights, 0, stop_policy="raise"), 0, 1000))
        self.assertTrue(490 <= samples.count('a') <= 510)
        self.assertEqual(len(samples), 1000)

    def test_sample_with_infinite_iter_same_as_loop(self):
        a = repeat('a')
        b = repeat('b')
        weights = {'a': 1, 'b': 1}
        # seed needs to be the same for this to work
        seed = 0
        repeating_samples = list(islice(sample({'a': a, 'b': b}, weights, seed, stop_policy="raise"), 1000))
        _a = ['a']
        _b = ['b']
        loop_samples = list(islice(sample({'a': _a, 'b': _b}, weights, seed, stop_policy="loop"), 1000))
        self.assertEqual(repeating_samples, loop_samples)

    def test_sample_with_raise_policy_can_raise(self):
        a = ['a'] * 5
        b = repeat('b')
        weights = {'a': 1, 'b': 1}
        with self.assertRaises(RuntimeError):
            list(islice(sample({'a': a, 'b': b}, weights, 0, stop_policy="raise"), 1000))

    def test_sample_with_skip_policy_just_drops(self):
        a = ['a'] * 5
        b = repeat('b')
        weights = {'a': 1, 'b': 1}
        result = list(islice(sample({'a': a, 'b': b}, weights, 0, stop_policy="skip"), 1000))
        self.assertEqual(len(result), 1000)
        self.assertEqual(result.count('a'), 5)

    def test_sample_with_stop_policy_can_stop(self):
        a = ['a'] * 5
        b = repeat('b')
        weights = {'a': 1, 'b': 1}
        result = list(islice(sample({'a': a, 'b': b}, weights, 0, stop_policy="stop"), 1000))
        self.assertEqual(result.count('a'), 5)
        self.assertLess(len(result), 100)

    def test_with_uneven_sampling(self):
        a = repeat('a')
        b = repeat('b')
        weights = {'a': 1, 'b': 2}
        result = list(islice(sample({'a': a, 'b': b}, weights, 0, stop_policy="raise"), 1000))
        self.assertTrue(300 <= result.count('a') <= 350)
        self.assertTrue(650 <= result.count('b') <= 750)


if __name__ == '__main__':
    unittest.main()
