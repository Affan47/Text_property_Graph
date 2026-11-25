"""Phase-1 screening is label-blind and cannot silently filter predictor sources."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.tpg_revision import github_audit_v1 as audit


class GithubAuditTests(unittest.TestCase):
    def test_unicode_cve_ids_and_case(self):
        r=audit.signals('cve\u20112024\u20114577 and CVE-2012-1823','CVE-2024-4577')
        self.assertEqual(r['id_group'],'target_and_other')
        self.assertFalse(r['other_only_ids'])
        self.assertTrue(r['multiple_ids'])

    def test_other_cve_is_a_signal_not_a_rejection(self):
        r=audit.signals('A bypass of CVE-2012-1823 using Windows best-fit conversion.','CVE-2024-4577')
        self.assertTrue(r['other_only_ids'])
        self.assertNotIn('decision',r)
        self.assertNotIn('accept',r)

    def test_absent_ids_and_limited_content_are_not_truth_labels(self):
        self.assertEqual(audit.signals('Fix the pip installation endpoint.','CVE-2024-21574')['id_group'],'no_id')
        r=audit.signals('The excerpt only shows repository metadata.','CVE-2024-27612')
        self.assertTrue(r['limited_content_cue'])
        self.assertNotIn('target',r)

    def test_sampling_deterministic_label_blind_and_training_only(self):
        cfg=audit.data.read(audit.CONFIG)
        rows=[{'cve':f'CVE-2024-{i:04d}','document_id':str(i),'split':'train',
               'signals':{'other_only_ids':i<5,'multiple_ids':5<=i<10,'limited_content_cue':10<=i<20}}
              for i in range(30)]
        selected=audit.select_sample(rows,cfg)
        self.assertEqual(len(selected),16)
        self.assertEqual(selected,audit.select_sample(list(reversed(rows)),cfg))
        tagged=[{**r,'target':i%2} for i,r in enumerate(rows)]
        self.assertEqual([r['cve'] for r in selected],[r['cve'] for r in audit.select_sample(tagged,cfg)])
        altered=deepcopy(rows);altered[0]['split']='validation'
        with self.assertRaises(ValueError):audit.select_sample(altered,cfg)

    def test_partitions_retain_every_cve_and_count_documents(self):
        rows=[];targets={}
        for split in ('train','validation'):
            for i in range(4):
                c=split+str(i);targets[c]=i%2;mask=int(i<2)
                rows.append({'cve':c,'split':split,'source_mask':[1,mask,0,0],
                             'documents':[{'source_text_type':'github_summary'}] if mask else []})
        table,by_class,parts=audit.coverage(rows,targets)
        for split in parts:
            self.assertFalse(set(parts[split]['1'])&set(parts[split]['0']))
            self.assertEqual(len(parts[split]['1'])+len(parts[split]['0']),4)
        self.assertEqual(sum(r['accepted_G_documents'] for r in table),4)
        self.assertTrue(all(r['listed_percent']==50 for r in table))
        self.assertTrue(all(r['coverage_percent']==50 for r in by_class))


if __name__=='__main__':unittest.main()
