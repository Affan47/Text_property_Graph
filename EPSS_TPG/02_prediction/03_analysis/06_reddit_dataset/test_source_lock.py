import itertools
import random
import re
import unittest

from lock_rd01_sources import normalize, disclosure, near_pairs


class SourceLockTests(unittest.TestCase):
    def test_normalization_retains_identifiers_versions(self):
        self.assertEqual(normalize('  CVE-2024-1234\n V2.3  '),'cve-2024-1234 v2.3')
        self.assertNotEqual(normalize('Version 2.3'),normalize('Version 2.4'))

    def test_disclosure_generic_vs_numeric(self):
        self.assertTrue(disclosure('EPSS score: 0.93')['quarantine'])
        self.assertTrue(disclosure('EPSS score: 0.93')['epss_numeric_context'])
        self.assertTrue(disclosure('We discuss EPSS')['quarantine'])
        self.assertFalse(disclosure('We discuss EPSS')['epss_numeric_context'])
        self.assertFalse(disclosure('CVSS: 9.8; CVE-2024-1234')['quarantine'])
        self.assertFalse(disclosure('EPSS details for CVE-2024-1234')['epss_numeric_context'])

    def test_catalogue_is_not_epss_disclosure(self):
        result=disclosure('Listed in VulnCheck KEV')
        self.assertFalse(result['quarantine']); self.assertTrue(result['catalogue_cue'])

    def test_exact_prefix_join_matches_bruteforce(self):
        rng=random.Random(17); base=[f'word{i}' for i in range(80)]
        texts={}
        for i in range(28):
            words=base[:]
            for _ in range(i%7): words[rng.randrange(len(words))]=f'new{rng.randrange(99)}'
            texts[str(i)]=' '.join(words[:80-i%4])
        for threshold in (.6,.9,1.):
            ss={k:{tuple(re.findall(r'\w+',v)[i:i+5]) for i in range(len(v.split())-4)} for k,v in texts.items()}
            expected={(min(a,b),max(a,b)) for a,b in itertools.combinations(texts,2) if len(ss[a]&ss[b])/len(ss[a]|ss[b])>=threshold}
            actual={(a,b) for a,b,_ in near_pairs(texts,threshold)}
            self.assertEqual(actual,expected)


if __name__=='__main__': unittest.main()
