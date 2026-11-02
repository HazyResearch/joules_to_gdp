"""Substantive checks for overlap, changing power, failures and conservation."""
import unittest
from active_request_energy import RequestInterval as R, attribute_batch


class AttributionTests(unittest.TestCase):
    def test_equal_overlaps(self):
        result = attribute_batch([R('a', 0, 90), R('b', 0, 90)], batch_start_s=0, batch_end_s=90, batch_energy_j=200)
        self.assertEqual(result.request_energy_j, {'a': 100., 'b': 100.})

    def test_tail_only_to_remaining_request(self):
        result = attribute_batch([R('a', 0, 90), R('b', 0, 180)], batch_start_s=0, batch_end_s=180, batch_energy_j=200)
        self.assertEqual(result.request_energy_j, {'a': 50., 'b': 150.})

    def test_power_change_at_completion(self):
        result = attribute_batch([R('a', 0, 5), R('b', 0, 10)], batch_start_s=0, batch_end_s=10, batch_energy_j=30, counter_times_s=[0, 5, 10], cumulative_energy_j=[100, 110, 130])
        self.assertEqual(result.request_energy_j, {'a': 5., 'b': 25.})
        self.assertEqual(result.method, 'active_request_metered_intervals')

    def test_staggered_start_and_idle_overhead(self):
        result = attribute_batch([R('a', 10, 30), R('b', 20, 40)], batch_start_s=0, batch_end_s=50, batch_energy_j=50)
        self.assertEqual(result.request_energy_j, {'a': 25., 'b': 25.})
        self.assertEqual(result.no_active_energy_j, 20.)

    def test_missing_coverage_or_mismatched_meter_fails(self):
        base = dict(requests=[R('a',0,10)],batch_start_s=0,batch_end_s=10,batch_energy_j=10)
        with self.assertRaises(ValueError):
            attribute_batch(**base,counter_times_s=[1,10],cumulative_energy_j=[0,10])
        with self.assertRaises(ValueError):
            attribute_batch(**base,counter_times_s=[0,10],cumulative_energy_j=[0,20])
        with self.assertRaises(ValueError):
            attribute_batch([R('missing',float('nan'),10)],batch_start_s=0,batch_end_s=10,batch_energy_j=10)

    def test_variable_batch_sizes_and_repair_attempts(self):
        # No dependence on batch size 20 and no removal of a failed attempt.
        for n in [1, 3, 20, 23]:
            result = attribute_batch([R(str(i),0,i+1) for i in range(n)],batch_start_s=0,batch_end_s=n,batch_energy_j=100)
            self.assertAlmostEqual(sum(result.request_energy_j.values()),100)
            self.assertTrue(all(v > 0 for v in result.request_energy_j.values()))
            self.assertAlmostEqual(sum(result.equal_split_energy_j.values()),100)


if __name__ == '__main__':
    unittest.main()
