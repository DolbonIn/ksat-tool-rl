"""All approval strings below are synthetic unit-test fixtures, never training evidence."""
import copy, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_and_export import ROOT,load,audit,same_answer
class Tests(unittest.TestCase):
    def fixture(self):
        p=load(ROOT/"legacy/problems_28.jsonl")[0];t=load(ROOT/"legacy/traces_28.jsonl")[0]
        p.update(problem_id="UNIT-ONLY",family_id="UNIT-FAMILY",split="train",intended_split="train")
        p["source"]["administration_year"]=2021
        p["input"]={"question":"3x+5=20에서 x를 구하시오.","passages":[],"choices":[],"assets":[],"modality":"text"}
        p["gold"]={"answer_type":"numeric","value":"5","choice_id":None,"source_ref":"UNIT-ONLY"}
        p["rights"].update(status="approved",evidence_ref="UNIT-ONLY")
        for c in p["checks"].values(): c.update(status="pass",report_ref="UNIT-ONLY")
        t.update(problem_id=p["problem_id"],trace_id="UNIT-TRACE")
        t["solution"].update(rationale="3x=15이므로 x=5이다.",final="정답: 5",answer_value="5",choice_id=None)
        t["generation"].update(gold_visible=False,finish_reason="stop",mode="no_tool")
        for c in t["checks"].values(): c.update(status="pass",report_ref="UNIT-ONLY")
        return p,t
    def test_legacy_zero(self):
        report,sft,rl=audit(load(ROOT/"legacy/problems_28.jsonl"),load(ROOT/"legacy/traces_28.jsonl"))
        self.assertEqual((report["traces"],len(sft),len(rl)),(28,0,0))
    def test_valid_fixture(self):
        p,t=self.fixture();_,sft,rl=audit([p],[t],"think_tags")
        self.assertEqual((len(sft),len(rl)),(1,1))
        self.assertIn("<think>",sft[0]["completion"][0]["content"])
    def test_missing(self):
        p,t=self.fixture();p["input"]["question"]=None
        _,sft,rl=audit([p],[t]);self.assertEqual((len(sft),len(rl)),(0,0))
    def test_holdout(self):
        p,t=self.fixture();p.update(split="test",intended_split="test")
        _,sft,rl=audit([p],[t]);self.assertEqual((len(sft),len(rl)),(0,0))
    def test_gold_visible(self):
        p,t=self.fixture();t["generation"]["gold_visible"]=True
        _,sft,rl=audit([p],[t]);self.assertEqual((len(sft),len(rl)),(0,1))
    def test_truncation(self):
        p,t=self.fixture();t["generation"]["finish_reason"]="length"
        _,sft,_=audit([p],[t]);self.assertFalse(sft)
    def test_family_leak(self):
        p,t=self.fixture();q=copy.deepcopy(p);q.update(problem_id="UNIT-2",split="test",intended_split="test")
        with self.assertRaises(ValueError): audit([p,q],[t])
    def test_exact_leak(self):
        p,t=self.fixture();q=copy.deepcopy(p);q.update(problem_id="UNIT-2",family_id="OTHER",split="test",intended_split="test")
        with self.assertRaises(ValueError): audit([p,q],[t])
    def test_wrong_answer(self):
        p,t=self.fixture();t["solution"]["answer_value"]="6"
        _,sft,_=audit([p],[t]);self.assertFalse(sft)
    def test_numeric(self):
        self.assertTrue(same_answer("0.5","1/2","numeric"))
        self.assertFalse(same_answer("sqrt(2)","sqrt(2)","numeric"))
    def test_prompt_excludes_gold(self):
        p,t=self.fixture();p["gold"]["source_ref"]="SECRET_LABEL"
        _,sft,_=audit([p],[t]);self.assertNotIn("SECRET_LABEL",str(sft[0]["prompt"]))
    def test_demo_block(self):
        p,t=self.fixture();p["demo_only"]=True
        _,sft,rl=audit([p],[t]);self.assertFalse(sft);self.assertFalse(rl)
if __name__=="__main__": unittest.main()
