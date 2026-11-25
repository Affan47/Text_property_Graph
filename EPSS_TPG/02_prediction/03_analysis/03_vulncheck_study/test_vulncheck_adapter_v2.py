"""VK02 synthetic policy, predictor separation, train-only fitting and reload tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'01_package'))
from epss.vulncheck_study import adapter_v2 as adapter, policy_v2 as policy
from epss.vulncheck_study import classifier_smoke_v2 as smoke


def row(cve='CVE-2025-10001',split='train',text='A vulnerability affects product software.'):
    return {'cve':cve,'split':split,'documents':[{'document_id':'description:fixture',
        'source_text_type':'description','text':text}], 'source_mask':[1,0,0,0],'structured':[.2,1.]}


class PolicyTests(unittest.TestCase):
    def test_reviewed_cases(self):
        for case in policy.review_cases():
            with self.subTest(case=case['case']):
                self.assertEqual(policy.inspect(case['text'],case['cve'],case['source'])['decision'],case['expected'])

    def test_function_exception_is_span_local(self):
        text='def vulncheck(ipaddress, port): pass\nVulnCheck reported this vulnerability.'
        r=policy.inspect(text,'CVE-2023-23333','smp')
        self.assertEqual(len(r['ignored_spans']),1)
        self.assertEqual(r['decision'],'quarantine')
        self.assertEqual(len(r['evidence_spans']),1)

    def test_function_call_is_not_silently_cleared(self):
        self.assertEqual(policy.inspect('VulnCheck(x)','CVE-2023-23333','smp')['decision'],'quarantine')

    def test_unicode_hyphens_preserve_spans_and_identity(self):
        text='CVE identifier is `CVE\u20112023\u201112345`'
        r=policy.inspect(text,'CVE-2019-1367','non_github_summary')
        self.assertIn('summary_explicit_other_target',r['reasons'])
        self.assertEqual(r['other_cve_ids'],['CVE-2023-12345'])
        span=r['evidence_spans'][0]
        self.assertEqual(text[span['start']:span['end']],text[:-1])
        r=policy.inspect('Known\u2011Exploited Vulnerabilities','CVE-2025-10001','smp')
        self.assertEqual(r['decision'],'quarantine')

    def test_explicit_target_vs_comparative_context(self):
        good='CVE-2019-1367 differs from CVE-2019-1221.'
        self.assertEqual(policy.inspect(good,'CVE-2019-1367','non_github_summary')['decision'],'accept')
        bad='CVE identifier is ```CVE-2023-12345```'
        self.assertEqual(policy.inspect(bad,'CVE-2019-1367','non_github_summary')['decision'],'quarantine')
        self.assertEqual(policy.inspect(bad,'CVE-2019-1367','smp')['decision'],'accept')

    def test_epss_disclosures_and_negated_catalogue_claims(self):
        for text in ('EPSS probability is 0.7','Not listed in KEV','Not listed in VulnCheck'):
            self.assertEqual(policy.inspect(text,'CVE-2025-10001','non_github_summary')['decision'],'quarantine')
        self.assertEqual(policy.inspect('An actively exploited bug.','CVE-2025-10001','description')['decision'],'accept')


class AdapterTests(unittest.TestCase):
    def setUp(self):self.p=adapter.config()

    def test_config_has_no_duplicate_keys(self):
        def pairs(values):
            keys=[k for k,v in values]
            self.assertEqual(len(keys),len(set(keys)))
            return dict(values)
        json.loads(adapter.CONFIG.read_text(),object_pairs_hook=pairs)

    def test_predictors_reject_labels_and_old_test(self):
        adapter.validate_predictor(row(),self.p,2)
        for name in ('target','in_vckev','epss_score','quarantine_count'):
            r=row();r[name]=1
            with self.assertRaises(ValueError):adapter.validate_predictor(r,self.p,2)
        with self.assertRaises(ValueError):adapter.validate_predictor(row(split='test'),self.p,2)

    def test_bad_mask_or_quarantined_document_rejected(self):
        r=row();r['source_mask']=[0,0,0,0]
        with self.assertRaises(ValueError):adapter.validate_predictor(r,self.p,2)
        with self.assertRaises(ValueError):adapter.validate_predictor(row(text='Listed in KEV.'),self.p,2)

    def test_optional_sources_and_document_dedup(self):
        raw={'description':'Text.','social_media_posts':[{'post':'Post text','platform':'Reddit'}]*2}
        docs=list(adapter.documents(raw,self.p))
        self.assertEqual(len(docs),2)
        self.assertEqual({r['source_text_type'] for r in docs},{'description','smp'})

    def test_selection_preserves_roles_and_needs_no_epss(self):
        records=[];manifest={}
        for i in range(40):
            c=f'CVE-2025-{10000+i}';role='train' if i<20 else 'validation' if i<36 else 'test'
            records.append({'cve':c,'in_vckev':i%3==0})
            manifest[c]={'split':role}
        p=copy.deepcopy(self.p);p['smoke']['counts']={'train':8,'validation':6}
        ids=adapter.select_ids(records,manifest,p,'smoke')
        self.assertEqual(ids,adapter.select_ids(records,manifest,p,'smoke'))
        self.assertTrue(all(manifest[c]['split']==s for s,cs in ids.items() for c in cs))
        self.assertEqual({s:len(cs) for s,cs in ids.items()},{'train':8,'validation':6})
        full=adapter.select_ids(records,manifest,p,'full')
        self.assertEqual(len(full['train']),20);self.assertEqual(len(full['validation']),16)

    def test_structured_preprocessing_is_training_only(self):
        from epss.consolidated.contracts import StructuredEncoder
        fields={'cvss_vector':'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H',
                'cvss_vector_version':'3.1','cvss_vector_score':9.8,'n_references_nvd':3}
        state=StructuredEncoder.fit([{'cve':'CVE-2025-10001','split':'train',**fields}])
        saved=copy.deepcopy(state)
        vector,issues=StructuredEncoder.transform(fields,state,['cvss','reference_count'])
        other={**fields,'n_references_nvd':9999}
        other_vector,_=StructuredEncoder.transform(other,state,['cvss','reference_count'])
        self.assertEqual(vector[-2],0.)
        self.assertNotEqual(vector[-2],other_vector[-2]);self.assertEqual(state,saved)
        with self.assertRaises(ValueError):StructuredEncoder.fit([{'cve':'x','split':'validation',**fields}])
        invalid={**fields,'cvss_vector_version':'2.0'}
        _,flags=StructuredEncoder.transform(invalid,state,['cvss','reference_count'])
        self.assertIn('version_field_disagrees_with_vector',flags)


class ClassifierTests(unittest.TestCase):
    def setUp(self):self.settings=adapter.config()['smoke']

    def test_validation_tokens_not_fitted(self):
        train=[row(text='ordinary product vulnerability'),row(cve='CVE-2025-10002',text='memory corruption')]
        vec=smoke.fit_vectorizer(train,self.settings)
        val=[row(split='validation',text='exclusivelyvalidationtoken')]
        self.assertNotIn('exclusivelyvalidationtoken',vec.vocabulary_)
        self.assertEqual(smoke.features(val,'description_tfidf',vec).nnz,0)
        with self.assertRaises(ValueError):smoke.fit_vectorizer(val,self.settings)

    def test_metadata_and_unselected_sources_not_features(self):
        rows=[row()];vec=smoke.fit_vectorizer(rows,self.settings)
        changed=copy.deepcopy(rows);changed[0]['cve']='CVE-2099-99999'
        changed[0]['documents'].append({'document_id':'smp:extra','source_text_type':'smp','text':'SPECIAL SOCIAL TEXT'})
        changed[0]['source_mask'][3]=1
        for mode in ('structured_only','description_tfidf','description_tfidf_structured'):
            np.testing.assert_array_equal(smoke.features(rows,mode,vec).toarray(),smoke.features(changed,mode,vec).toarray())
        self.assertNotEqual(smoke.features(rows,'source_masks_only').nnz,smoke.features(changed,'source_masks_only').nnz)

    def test_model_feature_builder_rejects_target_fields(self):
        r=row();r['target']=1
        with self.assertRaises(ValueError):smoke.features([r],'structured_only')

    def test_checkpoint_reload(self):
        rows=[row(cve=f'CVE-2025-{10000+i}',text='memory attack' if i%2 else 'product update') for i in range(6)]
        vec=smoke.fit_vectorizer(rows,self.settings);x=smoke.features(rows,'description_tfidf',vec)
        model=LogisticRegression(solver='liblinear',random_state=17).fit(x,[0,1,0,1,0,1])
        bundle={'model':model,'vectorizer':vec,'variant':'description_tfidf','feature_dimension':x.shape[1]}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.joblib';joblib.dump(bundle,path)
            np.testing.assert_array_equal(smoke.predict(bundle,rows),smoke.predict(joblib.load(path),rows))

    def test_undefined_metrics_and_calibration_accounting(self):
        m=smoke.score([0,0,1,1],[.1,.1,.1,.1])
        self.assertIsNone(m['precision']);self.assertEqual(m['recall'],0)
        self.assertEqual(m['confusion_matrix_true_rows_predicted_columns'],[[2,0],[2,0]])
        self.assertEqual(sum(r['n'] for r in m['calibration']),4)
        self.assertIsNone(smoke.score([0,0],[0,1])['roc_auc'])
        with self.assertRaises(ValueError):smoke.score([0,1],[.1,float('nan')])


if __name__=='__main__':unittest.main()
