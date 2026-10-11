"""Archive independent post-seal readback without changing the base evidence."""
from pathlib import Path
import datetime, hashlib, json, shutil
ROOT=Path('/workspace/Habitable-Moon-Project')
WORK=Path('/workspace')
BASE=ROOT/'analysis/genesis_volume_slip_validation/evidence'
DEST=ROOT/'analysis/genesis_volume_slip_validation/final_readback'
RESULT=WORK/'genesis_volume_slip_package_independent_readback_20261009.json'
FIRST=WORK/'genesis_volume_slip_package_independent_readback_capture_20261009'
FINAL=WORK/'genesis_volume_slip_package_independent_readback_capture_20261009_final02'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def raw(p):
 b=p.read_bytes();return {'bytes':len(b),'raw_sha256':hashlib.sha256(b).hexdigest()}
def load(p):return json.loads(p.read_text())
def sealed(obj):
 assert obj['sha256']==hashlib.sha256(canonical({k:v for k,v in obj.items() if k!='sha256'})).hexdigest()
 return obj
def write_sealed(p,v):
 v['sha256']=hashlib.sha256(canonical(v)).hexdigest();p.write_text(json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def copy(p,rel):
 target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert raw(p)==raw(target)
base_manifest=BASE/'package_manifest.json';base_pin=raw(base_manifest);base=sealed(load(base_manifest))
result=sealed(load(RESULT));first=sealed(load(FIRST/'execution_facts.json'));final=sealed(load(FINAL/'execution_facts.json'))
assert result['status']=='PASS' and result['all_checks_pass']
assert result['manifest']==base_pin and result['manifest_self_sha256']==base['sha256']
assert result['package_member_count']==base['member_count'] and result['package_member_bytes']==base['total_member_bytes']
assert first['actual_exit_code']==1 and final['actual_exit_code']==0
assert first['changed_sources']==final['changed_sources']==[]
assert final['stderr.log']['bytes']==0
for directory,facts in [(FIRST,first),(FINAL,final)]:
 for k in ['stdout.log','stderr.log']:assert raw(directory/k)==facts[k]
 for path,pin in facts['source_before'].items():assert raw(Path(path))==pin==facts['source_after'][path]
assert final['result.json']==raw(RESULT)
stdout=load(FINAL/'stdout.log');assert stdout['raw']==raw(RESULT) and stdout['self_sha256']==result['sha256']
assert not DEST.exists();DEST.mkdir()
for dirname,path in [('first_failure/capture',FIRST),('final02/capture',FINAL)]:
 for p in path.iterdir():assert p.is_file();copy(p,Path(dirname)/p.name)
for source,target in [
 ('genesis_volume_slip_package_independent_readback_20261009.py','initial_checker.py'),
 ('genesis_volume_slip_package_independent_readback_capture_20261009.py','initial_capture.py'),
 ('genesis_volume_slip_package_independent_readback_20261009_final02.py','final02_checker.py'),
 ('genesis_volume_slip_package_independent_readback_capture_20261009_final02.py','final02_capture.py')]:copy(WORK/source,'sources/'+target)
copy(RESULT,'final02/readback.json');copy(base_manifest,'base_package_manifest.json');copy(Path(__file__).resolve(),'packaging/archive_sidecar.py')
readme=f'''НЕЗАВИСИМАЯ ПРОВЕРКА ЗАКРЕПЛЁННОГО ПАКЕТА GENESIS VOLUME SLIP

Это отдельный новый sidecar после seal. Базовый evidence/ не менялся.
Базовый manifest: ../evidence/package_manifest.json
Raw SHA256: {base_pin['raw_sha256']}
Self SHA256: {base['sha256']}
Состав: {base['member_count']} members / {base['total_member_bytes']} bytes.
Его буквальная копия: base_package_manifest.json.

Первый readback завершился отказом самого внешнего checker на metadata:
readback/genesis_volume_slip_coupled_independent_final01_readback_20261009.json
имеет собственную первичную sha256 и reference report_sha256 независимого
coupled audit. Checker ошибочно распознал обе как первичные подписи и выдал
AssertionError ambiguous member selfseal. Первая проверка не объявлена PASS.
Actual exit={first['actual_exit_code']}, elapsed={first['actual_seconds']:.12f}s,
stderr={first['stderr.log']['bytes']} bytes, changed_sources=[].
Raw stdout/stderr/execution_facts сохранены в first_failure/capture/.
Первоначальный checker остаётся в источнике sources/initial_checker.py и в
НЕИЗМЕНЁННОМ базовом evidence/packaging/independent_readback.py.

Исправлен только внешний checker в НОВОМ файле: при наличии собственной
sha256 она является первичной self-seal; report_sha256 этого readback-объекта
остаётся ссылкой на проверяемый coupled report. Отчёты actual/material/coupled
при этом сами проверяются отдельно со своей report_sha256, exact captured
stdout и raw hashes входных файлов. Исправление не меняет научные уравнения,
сохранения, тесты или семь ранее успешных physics/test captures.

Fresh final02 independent readback PASS:
Actual exit={final['actual_exit_code']}, elapsed={final['actual_seconds']:.12f}s,
stderr={final['stderr.log']['bytes']} bytes, changed_sources=[].
Raw capture: final02/capture/. Полный self-sealed result: final02/readback.json.
Result self SHA256: {result['sha256']}
Проверены exact member inventory/bytes, {len(result['self_sealed_members'])} self-sealed JSON,
seven final captures со всеми snapshot/live pins и report/stdout/input bindings,
actual physics parity/direct-resumed bytes, {result['old_member_count']} прежних members,
native6, оба старых actual semantic replay captures и literal old-doc suffixes.
{result['tests_passed']} ранее выполненных тестов подтверждены по raw pytest capture.
Никаких повторных физических расчётов или тестов этот readback не выполнял.

Authentic actual-source restart относится к inactive продолжению Genesis.
Отдельный active coupled fixture — declared synthetic runtime VIEW; он не
сертифицирует authentic active restart или continuous physical velocity/force
trajectory. Interval-average constant-rate closure и TOP-face plane transport
остаются явно ограниченными. Calculated sharp banks/creation work, all-pair
full-time admission включая reverse C5->C1 и actual detailed underthrust path
ещё открыты. Пункт1 и исходные этапы2–6 не закрыты; автоматическая цель OFF.

Все исходные outside файлы сохранены. Этот sidecar имеет собственный manifest,
список raw member bytes/hashes и canonical self-seal excluding sha256.
В базовый evidence/ не добавлен ни один файл и не переписан ни один member.
'''
(DEST/'README.txt').write_text(readme)
verification={'format':'genesis-volume-slip-post-seal-readback-sidecar-verification-0.1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','base_manifest':base_pin,'base_manifest_self_sha256':base['sha256'],'base_package_members':base['member_count'],'base_package_member_bytes':base['total_member_bytes'],'base_evidence_unchanged':True,'readback_result':raw(RESULT),'readback_self_sha256':result['sha256'],'first_failure_metadata_only':True,'initial_checker_primary_vs_reference_signature_detection_corrected_in_external_new_file':True,'first_failure_exit':first['actual_exit_code'],'first_failure_seconds':first['actual_seconds'],'first_failure_stderr_bytes':first['stderr.log']['bytes'],'final02_exit':final['actual_exit_code'],'final02_seconds':final['actual_seconds'],'final02_stderr_bytes':0,'source_pins_unchanged_in_both_readback_runs':True,'new_physics_or_test_runs_performed':False,'all_original_outside_artifacts_preserved':True,'authentic_inactive_restart_exact':True,'synthetic_active_runtime_view_certified_as_authentic_restart':False,'synthetic_continuous_physical_velocity_force_trajectory_certified':False,'point_one_completed':False}
assert raw(base_manifest)==base_pin;write_sealed(DEST/'verification.json',verification)
files=[{'path':str(p.relative_to(DEST)),**raw(p)} for p in sorted(DEST.rglob('*')) if p.is_file()]
manifest={'format':'genesis-volume-slip-post-seal-readback-sidecar-package-0.1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_manifest':base_pin,'base_manifest_self_sha256':base['sha256'],'base_package_members':base['member_count'],'base_package_member_bytes':base['total_member_bytes'],'member_count':len(files),'total_member_bytes':sum(p['bytes'] for p in files),'files':files}
write_sealed(DEST/'package_manifest.json',manifest)
assert raw(base_manifest)==base_pin
print(json.dumps({'status':'PASS','sidecar':str(DEST),'member_count':manifest['member_count'],'total_member_bytes':manifest['total_member_bytes'],'manifest_raw':raw(DEST/'package_manifest.json'),'manifest_self_sha256':manifest['sha256'],'base_manifest_raw':base_pin},sort_keys=True))
