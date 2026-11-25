"""Full-run review checks using synthetic CVEs only."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_forecast_tpg_full as review
from test_forecast_tpg_review import with_graph_stats

task, cache, torch = review.task, review.cache, review.torch


class FullReviewTests(unittest.TestCase):
    def test_patience_ties_and_early_termination(self):
        config = {'epochs':5,'patience':2}
        rows = [{'validation_mae':x} for x in [.3,.2,.2,.25]]
        self.assertEqual(review.stopping_check(rows,config),'patience')
        with self.assertRaisesRegex(ValueError,'prematurely'):
            review.stopping_check(rows[:-1],config)
        with self.assertRaisesRegex(ValueError,'continues'):
            review.stopping_check(rows+[{'validation_mae':.1}],config)

    def test_paired_error_direction_and_ties(self):
        rows = [{'actual_day60_epss':.5,'a':.4,'b':.2},
                {'actual_day60_epss':.1,'a':.1,'b':.1}]
        p = review.paired(rows,'a','b')
        self.assertAlmostEqual(p['mae_difference_left_minus_right'],-.1)
        self.assertEqual((p['left_lower_error'],p['equal_error'],p['left_higher_error']),(1,1,0))
        with self.assertRaisesRegex(ValueError,'Empty comparison'):
            review.paired([],'a','b')

    def test_exact_predictions_have_no_error_share(self):
        rows = [{'cve':str(i),'actual_day60_epss':.1,'p':.1} for i in range(2)]
        stats = review.summarize_predictions(rows,'p')
        self.assertIsNone(stats['top5_share_absolute_error'])
        self.assertEqual(stats['sample_sd'],0)

    def test_synthetic_full_review_and_tamper(self):
        m,graphs,data = with_graph_stats()
        m['scale'] = 'full'
        config = task.config_for('full')
        config.update(epochs=2,patience=2)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d); source_dir=root/'raw';source_dir.mkdir()
                source=source_dir/'source.json';cache.write_json(source,[])
                prepared=root/'prepared';prepared.mkdir()
                graph_dir=root/'graphs';graph_dir.mkdir()
                m.update(source_pins={str(source):cache.sha256(source)},prepared_directory=str(prepared))
                data['preparation_summary.json']={'source_path':str(source)}
                cache.write_json(graph_dir/'manifest.json',m)
                cache.write_json(graph_dir/'cache_lock.json',{'files':{'manifest.json':cache.sha256(graph_dir/'manifest.json')}})
                output=root/'run'
                with patch.object(cache,'load_cache',return_value=(m,graphs,data)), \
                     patch.object(task,'config_for',return_value=config), \
                     redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                    task.run(graph_dir,output)
                    before={str(p):cache.sha256(p) for p in output.rglob('*') if p.is_file()}
                    # Real review must only restore and infer, never fit again.
                    with patch.object(task,'fit',side_effect=AssertionError('Review attempted training')):
                        result=review.review(output)
                    self.assertTrue(result['checks_passed'])
                    self.assertFalse(result['test_scoring_performed'])
                    self.assertEqual(len(result['results']),16)
                    self.assertEqual(len(result['paired_comparisons']),24)
                    self.assertEqual(before,{str(p):cache.sha256(p) for p in output.rglob('*') if p.is_file()})
                    self.assertIn('Learning Curves',review.explanation(result))
                    (output/'predictions.json').write_text('[]',encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'Changed run artifact'):
                        review.review(output)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
