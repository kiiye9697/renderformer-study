"""Serial authorized stages after existing data process exits; never publish."""
from pathlib import Path
import argparse,subprocess,json,time,os
import ctypes
from ctypes import wintypes
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
def alive(pid):
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:return False
    try:return kernel.WaitForSingleObject(handle,0)==0x102
    finally:kernel.CloseHandle(handle)
ROOT=Path(__file__).resolve().parents[2];WORK=Path('F:/RenderFormer-work/training-pilot')
PY='F:/RenderFormer-work/venv/Scripts/python.exe'
p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
status=WORK/'pipeline-status.json'
def record(stage,**kw):status.write_text(json.dumps({'stage':stage,'unix_time':time.time(),**kw},indent=2))
def run(script,*args):
    record(script)
    with (WORK/(Path(script).stem+'.log')).open('w') as log:
        subprocess.run([PY,str(ROOT/'scripts/training'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
record('waiting_for_data_process',pid=a.wait_pid)
while alive(a.wait_pid):time.sleep(5)
try:
    run('audit_dataset.py');run('cache_features.py');run('cached_capacity.py')
    capacity=json.loads((WORK/'cached-capacity.json').read_text())
    safe=[r for r in capacity if r['free_bytes']>=1024**3]
    assert safe,'No tested microbatch retains 1 GiB free GPU memory'
    chosen=max(safe,key=lambda r:r['warm_samples_per_second'])
    (WORK/'batch-choice.json').write_text(json.dumps({'choice':chosen,'rule':'Highest measured warm throughput among 4/8 with >=1GiB free GPU memory; no quality/hyperparameter search'},indent=2))
    target=WORK/'run-v1';run('train_head.py','--microbatch',str(chosen['microbatch']),'--output',str(target))
    result=json.loads((target/'result.json').read_text())
    if result['stop_reason']!='validation_plateau':
        record('validation_review_required_test_unopened',result=result)
    else:
        run('evaluate_final.py','--run',str(target));run('export_summary.py','--run',str(target))
        run('historical_regression.py','--run',str(target))
        run('render_dataset_batch.py',str(WORK/'showcase-v1'));run('predict_showcase.py','--run',str(target))
        record('complete_local_results_ready',publication=str(target/'publication'))
except Exception as exc:
    record('failed',error=str(exc));raise
