"""Synthetic checks for source masking, timing and target isolation."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'01_package'))
from epss import forecast_multisource as m

FIELDS = {'description':'description', 'github_summary':'gh',
          'non_github_summary':'web', 'smp':'social_media_posts'}


def example():
    return {'cve':'CVE-2024-12345', 'description':'A product has a buffer overflow.',
            'gh':None, 'web':'A vendor describes a fix.',
            'social_media_posts':[
                {'post':'A public post.', 'platform':'reddit', 'date':'2024-01-02T12:00:00Z'}]}


class SourceTests(unittest.TestCase):
    def test_partial_sources_keep_cve_and_masks(self):
        row = m.adapt_record(example(), '2024-01-31', FIELDS)
        self.assertEqual(row['source_presence_mask'], [1,0,1,1])
        self.assertEqual(m.model_input(row,list(m.SOURCES))['source_mask'], [1,0,1,1])
        self.assertEqual(m.model_input(row,list(m.SOURCES),'historical_verified')['documents'], [])

    def test_empty_sources_never_fabricate_text(self):
        raw = example()
        raw.update(description='', gh='  ', web=None, social_media_posts=[])
        row = m.adapt_record(raw, '2024-01-31', FIELDS)
        self.assertEqual(row['cve'],raw['cve'])
        self.assertEqual(m.model_input(row,list(m.SOURCES)),{'documents':[],'source_mask':[0,0,0,0]})

    def test_source_ablation_removes_documents_and_mask(self):
        row = m.adapt_record(example(),'2024-01-31',FIELDS)
        for sources in ([], ['description'], ['smp'], ['github_summary','non_github_summary']):
            payload = m.model_input(row,sources)
            self.assertTrue(all(d['source_text_type'] in sources for d in payload['documents']))
            self.assertTrue(all(not payload['source_mask'][i] for i,s in enumerate(m.SOURCES) if s not in sources))

    def test_same_day_future_missing_invalid_excluded(self):
        for value in ('2024-01-31','2024-02-01',None,'', 'not-a-date','2024-02-30'):
            raw = example()
            raw['social_media_posts'][0]['date'] = value
            row = m.adapt_record(raw,'2024-01-31',FIELDS)
            self.assertEqual(row['source_presence_mask'][-1],1)
            self.assertEqual(m.model_input(row,['smp'])['source_mask'],[0,0,0,0])

    def test_timezone_and_naive_dates_remain_unverified(self):
        row = m.stored_date('2024-01-31T00:30:00+02:00','2024-01-31')
        self.assertEqual(row['date_compared'],'2024-01-30')
        self.assertEqual(row['relation'],'before_anchor')
        self.assertFalse(row['historical_verified'])
        self.assertEqual(m.stored_date('2024-01-30T00:00:00','2024-01-31')['timezone'],'not_supplied')

    def test_duplicates_conflicts_and_no_earliest_date_shortcut(self):
        raw = example()
        raw['social_media_posts'] *= 2
        row = m.adapt_record(raw,'2024-01-31',FIELDS)
        posts = [d for d in row['documents'] if d['source_text_type']=='smp']
        self.assertEqual(len(posts),1)
        self.assertEqual(len(posts[0]['source_paths']),2)
        self.assertEqual(m.model_input(row,['smp'])['source_mask'][-1],1)
        raw['social_media_posts'] = deepcopy(raw['social_media_posts'])
        raw['social_media_posts'][1] = {**raw['social_media_posts'][1],'date':'2024-02-05'}
        row = m.adapt_record(raw,'2024-01-31',FIELDS)
        self.assertEqual(m.model_input(row,['smp'])['source_mask'][-1],0)

    def test_same_day_different_post_times_are_not_silently_merged(self):
        raw = example()
        raw['social_media_posts'].append(
            {**raw['social_media_posts'][0], 'date':'2024-01-02T13:00:00Z'})
        row = m.adapt_record(raw,'2024-01-31',FIELDS)
        self.assertEqual(m.model_input(row,['smp'])['source_mask'][-1],0)
        raw['social_media_posts'][1]['date'] = '2024-01-02T14:00:00+02:00'
        row = m.adapt_record(raw,'2024-01-31',FIELDS)
        self.assertEqual(m.model_input(row,['smp'])['source_mask'][-1],1)

    def test_no_structured_labels_or_future_scores_enter_adapter(self):
        raw = example()
        expected = m.adapt_record(raw,'2024-01-31',FIELDS)
        raw.update(in_vckev=True,vckev_date_added='2025-01-01',epss_scores={'future':0.99},
                   target_30d=1,cvss_score=10,all_sources_urls=['https://example.test/future'])
        self.assertEqual(expected,m.adapt_record(raw,'2024-01-31',FIELDS))
        payload = m.model_input(expected,list(m.SOURCES))
        self.assertEqual(set(payload),{'documents','source_mask'})
        self.assertTrue(all(set(d)=={'document_id','source_text_type','text'} for d in payload['documents']))

    def test_no_silent_field_fallback_or_type_coercion(self):
        raw = example()
        del raw['gh']
        with self.assertRaises(ValueError):
            m.adapt_record(raw,'2024-01-31',FIELDS)
        raw = example()
        raw['gh'] = ['not a summary string']
        with self.assertRaises(ValueError):
            m.adapt_record(raw,'2024-01-31',FIELDS)

    def test_deterministic_selection_and_no_test_role(self):
        ids = {'train':['CVE-2024-10000','CVE-2024-10001'], 'validation':['CVE-2024-20000']}
        counts = {'train':1, 'validation':1}
        self.assertEqual(m.select_preview(ids,counts,'fixed:'),m.select_preview(ids,counts,'fixed:'))
        with self.assertRaises(ValueError):
            m.select_preview({**ids,'test':['CVE-2025-10000']},counts,'fixed:')

    def test_preview_lifecycle_reproduction_and_tamper_rejection(self):
        p = m.read(m.CONFIG)
        p['source_fields'] = FIELDS
        raw = example()
        cve = raw['cve']
        identity = {'protocol':p, 'ids':{'train':[cve], 'validation':[]}, 'source_pins':{}}
        ctx = ({cve:raw}, {cve:'2024-01-31'}, identity)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'preview'
            with patch.object(m,'OUTPUT',output), patch.object(m,'context',return_value=ctx):
                m.prepare(check_only=True)
                self.assertFalse(output.exists())
                m.prepare()
                m.prepare(verify_only=True)
                with self.assertRaises(ValueError):
                    m.prepare()
                value = m.read(output/'model_inputs.json')
                value['rows'][0]['source_mask'] = [0,0,0,0]
                m.write(output/'model_inputs.json',value)
                with self.assertRaises(ValueError):
                    m.prepare(verify_only=True)


if __name__ == '__main__':
    unittest.main()
