"""Review aggregation, CSV and integrity guards using small synthetic fixtures."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import copy
import review_forecast_tpg_fusion_diagnostic_v2 as review


class FusionReviewTests(unittest.TestCase):
    def test_repeat_aggregation_keeps_negative_and_positive_changes(self):
        rows=[]
        for seed in review.dq.PROTOCOL['model_seeds']:
            for variant in review.dq.PROTOCOL['variants']:
                for delta in ([-.1,.2,.2] if variant.startswith('swap_') else [0.]):
                    rows.append({'seed':seed,'variant':variant,'subset':'all_validation','cves':2,
                                 'original_mae':.3,'mean_absolute_prediction_change':.1,
                                 'mae_change_against_recipient_targets':delta})
        results=review.aggregate(rows)
        row=next(r for r in results if r['variant']=='swap_graph')
        self.assertAlmostEqual(row['mean_recipient_mae_change'],.1)
        self.assertEqual(row['repeats_increasing_error'],2)
        self.assertEqual(row['minimum_recipient_mae_change'],-.1)
        with self.assertRaisesRegex(ValueError,'Missing or duplicate'):
            review.aggregate(rows[:-1])

    def test_csv_nested_and_null_roundtrip(self):
        rows=[{'nested':{'z':1,'a':[2]},'empty':None,'truth':True,'value':.01}]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'table.csv'
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader()
                writer.writerow({'nested':json.dumps(rows[0]['nested'],sort_keys=True),'empty':'','truth':True,'value':.01})
            review.check_csv(path,rows)
            rows[0]['value']=.02
            with self.assertRaisesRegex(ValueError,'CSV content'):
                review.check_csv(path,rows)

    def test_semantic_concern_requires_the_actual_source_phrase(self):
        edge={'cve':'CVE-2022-1418','relation':'SEC_HAS_VERSION','source_text':'WordPress','target_text':'2.2.9',
              'evidence':{'text':'The Social Stickers WordPress plugin through 2.2.9 does not have CSRF checks.'}}
        original=json.dumps(edge,sort_keys=True)
        result=review.semantic_observations([edge])
        self.assertEqual(result[0]['assessment'],'likely_wrong_product_attachment')
        self.assertEqual(json.dumps(edge,sort_keys=True),original)
        edge['evidence']['text']='Different source wording.'
        with self.assertRaisesRegex(ValueError,'Version evidence differs'):
            review.semantic_observations([edge])

    def test_canonicalization_ignores_only_reconstruction_numbering(self):
        a=[{'split':'train','cve':'synthetic','nodes':3,'edges':2,
            'entities':[{'reconstructed_node_id':1,'cached_node_index':1,'text':'A','start':0,'end':1},
                        {'reconstructed_node_id':2,'cached_node_index':2,'text':'B','start':2,'end':3}],
            'structure_alignment':{'node_index_to_cached_index':[0,1,2],
                                  'edge_index_to_cached_index':[0,1],
                                  'renumbered_entity_nodes':0,'reordered_edges':0,'mode':'exact'}}]
        b=copy.deepcopy(a)
        b[0]['structure_alignment'].update(node_index_to_cached_index=[0,2,1],
            edge_index_to_cached_index=[1,0],renumbered_entity_nodes=2,reordered_edges=2,mode='entity_permutation')
        b[0]['entities'][0]['reconstructed_node_id']=2
        b[0]['entities'][1]['reconstructed_node_id']=1
        before=copy.deepcopy(b)
        self.assertEqual(review.canonical_table('security_coverage',a),
                         review.canonical_table('security_coverage',b))
        self.assertEqual(b,before)
        b[0]['entities'][0]['text']='tampered'
        self.assertNotEqual(review.canonical_table('security_coverage',a),
                            review.canonical_table('security_coverage',b))
        b[0]['structure_alignment']['node_index_to_cached_index']=[0,1,1]
        with self.assertRaisesRegex(ValueError,'bijection'):
            review.canonical_table('security_coverage',b)

    def test_edge_canonicalization_preserves_direction_and_qualifiers(self):
        a=[{'split':'validation','cve':'synthetic','edge_id':3,'reconstructed_edge_id':7,
            'source_text':'A','target_text':'B','evidence':{'polarity':'negated'}}]
        b=copy.deepcopy(a); b[0]['reconstructed_edge_id']=20
        self.assertEqual(review.canonical_table('security_edge_cases',a),
                         review.canonical_table('security_edge_cases',b))
        b[0]['evidence']['polarity']='affirmed'
        self.assertNotEqual(review.canonical_table('security_edge_cases',a),
                            review.canonical_table('security_edge_cases',b))

    def test_lock_checks_exact_set_and_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in review.FILES:(root/name).write_text('{}',encoding='utf-8')
            lock={'schema':review.dq.VERSION,'test_scoring_performed':False,
                  'files':{name:review.sha256(root/name) for name in review.FILES}}
            review.write_json(root/'analysis_lock.json',lock)
            self.assertEqual(review.locked(root),lock)
            (root/'analysis.txt').write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Changed diagnostic artifact'):
                review.locked(root)


if __name__=='__main__':
    unittest.main()

