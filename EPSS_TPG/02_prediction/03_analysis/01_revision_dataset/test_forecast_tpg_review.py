"""Synthetic smoke-run review, coverage and artifact-preservation checks."""
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import review_forecast_tpg as review
from test_forecast_tpg import fixture

task, cache, torch = review.task, review.cache, review.torch


def with_graph_stats():
    m, graphs, data = fixture()
    node_names = {v:k for k,v in m['type_vocabulary']['node_types'].items()}
    edge_names = {v:k for k,v in m['type_vocabulary']['edge_types'].items()}
    m['graphs'] = []
    total = Counter()
    for c,g in graphs.items():
        nodes = Counter(node_names[i] for i in g['x'][:,:len(node_names)].argmax(1).tolist())
        edges = Counter(edge_names[i] for i in g['edge_type'].tolist())
        total.update(edges)
        m['graphs'].append({'cve':c,'embedded_nodes':len(g['x']),'unembedded_nodes':0,
                           'embedded_tokens':len(g['x']), 'stats':{'node_types':dict(nodes),'edge_types':dict(edges)}})
    m['edge_counts'] = dict(total)
    return m,graphs,data


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_graph_counts_and_same_token_baseline(self):
        m,graphs,_=with_graph_stats()
        rows=review.graph_findings(m,graphs)
        self.assertEqual([r['total_nodes'] for r in rows],[6,6])
        self.assertEqual([r['cves_with_typed_security_edges'] for r in rows],[2,2])
        self.assertEqual(sum(r['total_edges'] for r in rows),8)
        altered=copy.deepcopy(graphs)
        altered[m['ids']['train'][0]]['text_embedding'] += 1
        with self.assertRaisesRegex(ValueError,'baseline vectors differ'):
            review.graph_findings(m,altered)
        m['graphs'][0]['embedded_nodes']=0
        with self.assertRaisesRegex(ValueError,'coverage differs'):
            review.graph_findings(m,graphs)

    def test_history_checks_selection_ties_and_finite_values(self):
        rows=[{'epoch':i,'training_huber_loss':.1,'training_mae':.2,'validation_mae':.3} for i in (1,2)]
        self.assertEqual(review.check_history(rows,{'selected_epoch':1},{'epochs':3})['epoch'],1)
        with self.assertRaisesRegex(ValueError,'selection differs'):
            review.check_history(rows,{'selected_epoch':2},{'epochs':3})
        rows[0]['training_huber_loss']=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid recorded'):
            review.check_history(rows,{'selected_epoch':1},{'epochs':3})

    def test_no_missing_tokens_needs_no_source_parser(self):
        m,graphs,data=with_graph_stats()
        self.assertEqual(review.missing_token_cases(m,graphs,data),[])

    def test_full_synthetic_review_preserves_artifacts_and_rejects_tamper(self):
        m,graphs,data=with_graph_stats()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source_dir=root/'raw';source_dir.mkdir()
            source=source_dir/'source.json';cache.write_json(source,[])
            prepared=root/'prepared';prepared.mkdir()
            graph_dir=root/'graphs';graph_dir.mkdir()
            m.update(source_pins={str(source):cache.sha256(source)},prepared_directory=str(prepared))
            data['preparation_summary.json']={'source_path':str(source)}
            cache.write_json(graph_dir/'manifest.json',m)
            cache.write_json(graph_dir/'cache_lock.json',{'files':{'manifest.json':cache.sha256(graph_dir/'manifest.json')}})
            output=root/'run'
            with patch.object(cache,'load_cache',return_value=(m,graphs,data)), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                task.run(graph_dir,output)
                before={str(p):cache.sha256(p) for p in output.rglob('*') if p.is_file()}
                result=review.review(output)
                self.assertTrue(result['checks_passed'])
                self.assertFalse(result['training_performed'])
                self.assertFalse(result['test_scoring_performed'])
                self.assertEqual(before,{str(p):cache.sha256(p) for p in output.rglob('*') if p.is_file()})
                self.assertEqual(len(result['results']),6)
                self.assertIn('Matched Results',review.explanation(result))
                (output/'predictions.json').write_text('[]',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Changed run artifact'):
                    review.review(output)


if __name__=='__main__':
    unittest.main()
