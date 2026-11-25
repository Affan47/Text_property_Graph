from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recover_collectors as search
import repair_provenance as p


def empty():
    return {'data':[],'_meta':{'index':'vulncheck-kev','total_documents':0,'total_pages':0}}


class RecoveryTests(unittest.TestCase):
    def test_errors_never_become_negative(self):
        for http in (401,403,404,429,500,503,302):
            self.assertIsNone(p.classify_cve('CVE-2022-46908',http,empty())[0])

    def test_verified_empty(self):
        self.assertEqual(p.classify_cve('CVE-2022-46908',200,empty())[:2],(False,'SUCCESS_NOT_LISTED'))

    def test_ambiguous_empty_and_malformed(self):
        for body in (None,[],{}, {'data':[]}, {'data':[],'_meta':{}}, {'data':[], 'error':'not authorized'}):
            self.assertIsNone(p.classify_cve('CVE-2022-46908',200,body)[0])
        for k,v in (('truncated',True),('next_cursor','next'),('warnings',['partial']),('total_documents',1),('index','cisa-kev')):
            body=empty();body['_meta'][k]=v
            self.assertIsNone(p.classify_cve('CVE-2022-46908',200,body)[0])

    def test_match_required_for_positive(self):
        body=empty();body['data']=[{'cve':['CVE-2022-46908']}]
        self.assertTrue(p.classify_cve('CVE-2022-46908',200,body)[0])
        self.assertIsNone(p.classify_cve('CVE-2022-12345',200,body)[0])

    def test_timezone_and_raw_fields(self):
        raw={'id':'abc123','title':'CVE-2022-46908 example','selftext':'Raw\nbody',
             'created_utc':1670803200,'edited':False,'retrieved_on':1670806800}
        r=p.normalize_post(raw,'sha','uri','revision',1)
        self.assertEqual(r['raw_body'],'Raw\nbody')
        self.assertEqual(r['published_at_utc'],datetime.fromtimestamp(1670803200,tz=timezone.utc).isoformat())
        self.assertEqual(r['raw_record'],raw)
        self.assertFalse(r['historical_text_verified'])
        self.assertIsNone(r['edited_at_utc'])

    def test_naive_timestamp_not_guessed(self):
        r=p.normalize_post({'title':'CVE-2022-46908','timestamp':'2022-12-12T10:00:00'},'sha','uri','rev',1)
        self.assertIsNone(r['published_at_utc'])
        self.assertIsNone(r['post_id'])
        self.assertIn('MISSING_OR_INVALID_CREATED_UTC',r['quality_flags'])

    def test_permalink_id_conflict(self):
        r=p.normalize_post({'id':'abc','permalink':'/r/test/comments/def/post/','created_utc':10},'sha','uri','rev',1)
        self.assertIn('ID_PERMALINK_CONFLICT',r['quality_flags'])

    def test_capture_before_publication(self):
        r=p.normalize_post({'id':'abc','created_utc':20,'retrieved_on':10},'sha','uri','rev',1)
        self.assertIn('ARCHIVE_CAPTURE_BEFORE_CREATION',r['quality_flags'])

    def test_no_fake_epoch(self):
        for value in (True,False,'1234',float('nan'),float('inf'),-1):
            self.assertIsNone(p.epoch_utc(value))

    def test_exact_bytes_and_line_endings(self):
        raw=b'print(1)\r\n'
        self.assertEqual(search.candidates(raw)['git_bytes'],raw)
        self.assertEqual(search.candidates(raw)['lf'],b'print(1)\n')

    def test_raw_versions_not_deduplicated(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp); src=root/'raw.json';out=root/'out'
            src.write_text(json.dumps([{'id':'abc','created_utc':10,'title':'CVE-2022-46908','selftext':text}
                                       for text in ('first','edited','edited')]))
            p.preserve_posts(src,out,'local-test','fixture')
            self.assertEqual(p.read(out/'summary.json')['cve_containing_rows'],3)
            self.assertEqual(p.read(out/'summary.json')['post_ids_with_multiple_text_versions'],1)
            p.preserve_posts(src,out,'local-test','fixture',True)

    def test_kev_receipts_replay_and_tamper(self):
        class Response:
            status_code=200
            content=json.dumps(empty()).encode()
        class Session:
            def get(self,*args,**kwargs):return Response()
            def close(self):pass
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp); src=root/'source.json';out=root/'out'
            src.write_text('[{"cve":"CVE-2022-46908","in_vckev":false}]')
            with patch.dict('os.environ',{'VC_API_KEY':'test-not-a-real-key'}),patch.object(p.requests,'Session',Session),patch.object(p.time,'sleep'):
                p.verify_kev(src,out,limit=1,allow_network=True)
            p.verify_kev(src,out,limit=1,verify_only=True)
            response=next((out/'receipts').rglob('*.response.json'))
            response.write_text('{}')
            with self.assertRaises(ValueError):
                p.verify_kev(src,out,limit=1,verify_only=True)


if __name__=='__main__':
    unittest.main()
