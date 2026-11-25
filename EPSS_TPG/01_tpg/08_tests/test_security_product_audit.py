"""Preview guards and a two-record end-to-end test, without real-data training."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'02_examples/01_scripts'))
import audit_security_product_versions as audit


class PreviewTests(unittest.TestCase):
    def fixture(self):
        data={'task_cohorts.json':{'endpoint':{'train':['CVE-EXAMPLE-A'],'validation':['CVE-EXAMPLE-B']}},
              'split_manifest.json':{'CVE-EXAMPLE-A':{'split':'train','group_id':'a'},
                                     'CVE-EXAMPLE-B':{'split':'validation','group_id':'b'}},
              'labels.json':{c:{'eligible':{'endpoint':True}} for c in ('CVE-EXAMPLE-A','CVE-EXAMPLE-B')}}
        manifest={'ids':data['task_cohorts.json']['endpoint']}
        review={'validation_semantic_concerns':[],'semantic_observations':[]}
        return manifest,data,review

    def test_test_cases_rejected(self):
        m,d,r=self.fixture()
        r['semantic_observations']=[{'cve':'CVE-TEST'}]
        with self.assertRaisesRegex(ValueError,'outside development'):
            audit.select_examples(m,d,r,'smoke')

    def test_review_examples_deduplicated_not_silently_new_sample(self):
        m,d,r=self.fixture()
        r['semantic_observations']=[{'cve':'CVE-EXAMPLE-A'},{'cve':'CVE-EXAMPLE-A'}]
        sample,reviewed,ids,roles=audit.select_examples(m,d,r,'smoke')
        self.assertEqual(reviewed,['CVE-EXAMPLE-A'])
        self.assertEqual(len(ids),2)
        self.assertEqual(sample,m['ids'])

    def test_preview_writes_separate_graphs_and_locked_reports(self):
        m,d,r=self.fixture()
        pipeline=audit.SecurityPipeline(include_security_relations=True)
        pipeline_digest=hashlib.sha256(pipeline.frontend.nlp.to_bytes()).hexdigest()
        raw=[{'cve':c,'description':'The Example Dashboard WordPress plugin before 1.2.3 is affected.'}
             for c in ('CVE-EXAMPLE-A','CVE-EXAMPLE-B')]
        graphs={}
        for record in raw:
            graph=pipeline.run(record['description'],doc_id=record['cve'])
            p=pipeline.export_pyg(graph,embedding_dim=4,use_security_edge_types=True)
            graphs[record['cve']]={k:audit.torch.tensor(p[k],dtype=audit.torch.long if k in ('edge_index','edge_type') else audit.torch.float32)
                                  for k in ('x','edge_index','edge_type','edge_features')}
            graphs[record['cve']]['description_sha256']=audit.cache.text_digest(record['description'])
        m.update(type_vocabulary=p['type_vocabulary'],embedding_dim=4,
                 resources={'spacy_pipeline_sha256':pipeline_digest})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'raw').mkdir()
            source=root/'raw/source.json';audit.write_json(source,raw)
            d['preparation_summary.json']={'source_path':str(source)}
            m['source_pins']={str(source):audit.sha256(source)}
            for dirname,filename in [('review','analysis.json'),('diagnostic','analysis_lock.json'),
                                     ('cache/description_full_v1','cache_lock.json')]:
                target=root/dirname;target.mkdir(parents=True)
                audit.write_json(target/filename,{})
            output=root/'preview'
            with patch.object(audit.cache,'load_cache',return_value=(m,graphs,d)), \
                 patch.object(audit,'verify_review',return_value=r), \
                 patch.object(audit,'REVIEW',root/'review'),patch.object(audit,'DIAGNOSTIC',root/'diagnostic'), \
                 patch.object(audit.cache,'CACHE_ROOT',root/'cache'), \
                 patch.object(audit.frozen.task,'fit',side_effect=AssertionError('Preview attempted training')), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                audit.run('smoke',output,check_only=True)
                self.assertFalse(output.exists())
                audit.run('smoke',output)
                summary=audit.read_json(output/'analysis.json')
                self.assertFalse(summary['model_ready_cache'])
                self.assertFalse(summary['training'])
                self.assertFalse(summary['test_graphs'])
                self.assertEqual(len(list((output/'graphs').glob('*.json'))),2)
                comparisons=audit.read_json(output/'comparisons.json')
                self.assertEqual(comparisons[0]['after']['typed_security_relations'][0]['source'],
                                 'Example Dashboard WordPress plugin')
                self.assertTrue(all(audit.sha256(output/name)==digest for name,digest in
                                    audit.read_json(output/'preview_lock.json')['files'].items()))
                with self.assertRaisesRegex(ValueError,'already exists'):
                    audit.run('smoke',output)
                with self.assertRaisesRegex(ValueError,'smoke preview first'):
                    audit.run('full',root/'full')
                audit.verify_smoke(output,summary['pinned_files'])
                (output/'analysis.txt').write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Changed smoke artifact'):
                    audit.verify_smoke(output,summary['pinned_files'])
            self.assertEqual(m['source_pins'][str(source)],audit.sha256(source))


if __name__=='__main__':
    unittest.main()
