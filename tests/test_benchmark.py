from statepatch.benchmark import make_statepatchbench, read_manifest

def test_bench(tmp_path):
    items=make_statepatchbench(tmp_path,2)
    assert len(items)==16
    rows=read_manifest(tmp_path/'statepatchbench.jsonl')
    assert len(rows)==16
    assert (tmp_path/rows[0]['image']).exists()
