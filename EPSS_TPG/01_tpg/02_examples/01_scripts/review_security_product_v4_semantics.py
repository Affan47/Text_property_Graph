"""Layout of the private DQ14 ledger; its annotation reader is not distributed."""
import review_security_product_v4 as frozen

old = frozen.old
ROOT = old.TPG / "09_annotation/08_v4_semantic_review"
DEFAULT_OUTPUT = ROOT / "bounded_review_v1"

def run(*args, **kwargs):
    raise RuntimeError("DQ14 requires the original private review ledger and its matching reader.")

if __name__ == "__main__":
    run()
