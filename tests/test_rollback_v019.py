import numpy as np
from tectonics.rollback import RollbackParameters, zone_rollback_rate
from tectonics.subduction_memory import SlabZone


def z(**kw):
    d=dict(subducting_plate=0,overriding_plate=1,active=True,active_age_myr=100,slab_length_km=1500,slab_depth_km=950,buoyancy_factor=1.0,trench_length_km=1000)
    d.update(kw);return SlabZone(**d)

def test_young_shallow_zone_does_not_rollback():
    p=RollbackParameters(); assert zone_rollback_rate(z(active_age_myr=8,slab_depth_km=200),p)==0

def test_mature_deep_zone_rolls_back_but_is_capped():
    p=RollbackParameters(max_rollback_rate_km_per_myr=4.0)
    r=zone_rollback_rate(z(),p); assert 3.0 < r <= 4.5

def test_buoyancy_strengthens_rollback_smoothly():
    p=RollbackParameters(); assert zone_rollback_rate(z(buoyancy_factor=1.3),p)>zone_rollback_rate(z(buoyancy_factor=.7),p)

def test_inactive_slab_has_no_active_rollback():
    p=RollbackParameters(); assert zone_rollback_rate(z(active=False),p)==0
